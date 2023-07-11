import torch
import numpy as np
from sklearn import metrics
import tqdm
import math
from torchvision.utils import make_grid
from utils import transform
from utils import evaluation
from sklearn.metrics import average_precision_score
from utils.openmax import compute_train_score_and_mavs_and_dists,fit_weibull,openmax,Evaluation,get_correct_slot

class OSREvaluator:
    def __init__(self,num_known_classes=7,use_softmax=False):
        super(OSREvaluator, self).__init__()
        self.best_val_loss=math.inf
        self.use_softmax=use_softmax
        self.num_known_classes=num_known_classes

    @torch.no_grad()
    def postprocess(self, model, img,class_label=None):
        if class_label is not None:
            slots,scores,matching_loss= model(img,class_label)
        else:
            slots, scores=model(img)
            matching_loss=math.inf
        #num_class=scores.shape[-1]
        if self.use_softmax:
            scores = torch.softmax(scores, dim=-1)
        conf, pred = torch.max(scores.flatten(1,2), dim=-1)
        pred=pred%self.num_known_classes
        return slots,pred, conf,matching_loss

    def eval(self, model,test_loader, out_loader,epoch_idx,writer):
        model.eval()
        _pred_k, _pred_u, _labels,k_pred= [], [], [],[]
        correct=0
        val_loss=0.
        with torch.no_grad():
            for batch_idx,sample in enumerate(test_loader):
                with torch.set_grad_enabled(False):
                    img = sample['img'].cuda()
                    label = sample['label'].cuda()
                    class_label=sample['class_label'].cuda()
                    #__, __, height, width = img.shape
                    slots,pred, conf,matching_loss = self.postprocess(model, img,class_label)
                    #print(slots.shape)
                    val_loss+=matching_loss.item()
                    correct += pred.eq(label.data).sum().item()
                    k_pred.append(pred.data.cpu().numpy())
                    _pred_k.append(conf.data.cpu().numpy())
                    _labels.append(label.data.cpu().numpy())

            for batch_idx,sample in enumerate(out_loader):
                with torch.set_grad_enabled(False):
                    img = sample['img'].cuda()
                    #label = sample['label'].cuda()
                    slot_mask,pred, conf,__= self.postprocess(model, img)
                    _pred_u.append(pred.data.cpu().numpy())

        k_pred= np.concatenate(k_pred, 0)
        _pred_k = np.concatenate(_pred_k, 0)
        _pred_u = np.concatenate(_pred_u, 0)
        _labels = np.concatenate(_labels, 0)
        acc = correct / len(test_loader.dataset)
        val_loss /= len(test_loader.dataset)
        writer.add_scalar('Val/mse', val_loss, epoch_idx)
        results = evaluation.metric_ood(_pred_k,_pred_u)['Bas']

        # OSCR
        _oscr_socre = evaluation.compute_oscr(k_pred,_pred_k, _pred_u, _labels)

        # Average precision
        ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                           list(-_pred_k) + list(-_pred_u))

        results['ACC'] = acc*100
        results['OSCR'] = _oscr_socre * 100.
        results['AUPR'] = ap_score * 100
        print("Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(epoch_idx,
                                                                                          results['ACC'],
                                                                                          results['AUROC'],
                                                                                          results['OSCR']))
        writer.add_scalar('test/ACC', results['ACC'], epoch_idx)
        writer.add_scalar('test/OSCR', results['OSCR'], epoch_idx)
        writer.add_scalar('test/AUPR', results['AUPR'], epoch_idx)
        return results
        #test_visz_result=self.slot_visualization(val_list)
        #writer.add_image('eval/train',test_visz_result,epoch_idx)

    # def slot_visualization(self,val_list):
    #     result=[]
    #     for i in range(val_list):
    #         img=val_list[i][0]
    #         slot_mask=val_list[i][1]
    #         visz_list = []
    #         num_slot = slot_mask.shape[0]
    #         img = self.denormalizer(img)
    #         visz_list.append(img)
    #         for j in range(num_slot):
    #             attention = slot_mask[j].unsqueeze(0).repeat(3, 1, 1)
    #             visz_list.append(attention * img)
    #         visz_list = torch.stack(visz_list, 0)
    #         result.append(visz_list)
    #     result=torch.stack(result,1)
    #     return result

    def openmax_processor(self,train_loader,test_loader,out_loader,model):
        model.eval()

        scores, labels = [], []
        with torch.no_grad():
            # for batch_idx, sample in enumerate(test_loader):
            #     img = sample['img'].cuda()
            #     target = sample['label'].cuda()
            #     class_label = sample['class_label'].cuda()
            #     __, outputs, matching_loss=model(img,class_label)
            #     #print(outputs.shape)
            #     single_slot_score=get_correct_slot(outputs)
            #     #print(single_slot_score.shape)
            #     scores.append(single_slot_score)
            #     labels.append(target)

            for batch_idx, sample in enumerate(out_loader):
                img = sample['img'].cuda()
                target = sample['label'].cuda()
                __, outputs=model(img)
                single_slot_score=get_correct_slot(outputs)
                #print(single_slot_score.shape)
                scores.append(single_slot_score)
                labels.append(target)

        # Get the prdict results.
        scores = torch.cat(scores, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        scores = np.array(scores)[:, np.newaxis, :]
        labels = np.array(labels)

        # Fit the weibull distribution from training data.
        print("Fittting Weibull distribution...")



        _, mavs, dists = compute_train_score_and_mavs_and_dists(self.num_known_classes, train_loader, model)
        categories = list(range(0, self.num_known_classes))
        weibull_model = fit_weibull(mavs, dists, categories, 20, "euclidean")

        pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
        score_softmax, score_openmax = [], []
        for score in scores:
            so, ss = openmax(weibull_model, categories, score,
                             0.5, 3, "euclidean")  # openmax_prob, softmax_prob
            pred_softmax.append(np.argmax(ss))
            pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= 0.9 else self.num_known_classes)
            pred_openmax.append(np.argmax(so) if np.max(so) >= 0.9 else self.num_known_classes)
            score_softmax.append(ss)
            score_openmax.append(so)

        print("Evaluation...")
        eval_softmax = Evaluation(pred_softmax, labels, score_softmax)
        eval_softmax_threshold = Evaluation(pred_softmax_threshold, labels, score_softmax)
        eval_openmax = Evaluation(pred_openmax, labels, score_openmax)

        print(f"Softmax accuracy is %.3f" % (eval_softmax.accuracy))
        print(f"Softmax F1 is %.3f" % (eval_softmax.f1_measure))
        print(f"Softmax f1_macro is %.3f" % (eval_softmax.f1_macro))
        print(f"Softmax f1_macro_weighted is %.3f" % (eval_softmax.f1_macro_weighted))
        print(f"Softmax area_under_roc is %.3f" % (eval_softmax.area_under_roc))
        print(f"_________________________________________")

        print(f"SoftmaxThreshold accuracy is %.3f" % (eval_softmax_threshold.accuracy))
        print(f"SoftmaxThreshold F1 is %.3f" % (eval_softmax_threshold.f1_measure))
        print(f"SoftmaxThreshold f1_macro is %.3f" % (eval_softmax_threshold.f1_macro))
        print(f"SoftmaxThreshold f1_macro_weighted is %.3f" % (eval_softmax_threshold.f1_macro_weighted))
        print(f"SoftmaxThreshold area_under_roc is %.3f" % (eval_softmax_threshold.area_under_roc))
        print(f"_________________________________________")

        print(f"OpenMax accuracy is %.3f" % (eval_openmax.accuracy))
        print(f"OpenMax F1 is %.3f" % (eval_openmax.f1_measure))
        print(f"OpenMax f1_macro is %.3f" % (eval_openmax.f1_macro))
        print(f"OpenMax f1_macro_weighted is %.3f" % (eval_openmax.f1_macro_weighted))
        print(f"OpenMax area_under_roc is %.3f" % (eval_openmax.area_under_roc))
        print(f"_________________________________________")