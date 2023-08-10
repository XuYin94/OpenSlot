import torch
import numpy as np
from sklearn import metrics
import tqdm
import math
from utils.utils import get_correct_slot,multi_correct_slot_2,multi_correct_slot,slot_max,slot_score,log_visualizations
from utils import evaluation
from sklearn.metrics import average_precision_score
from utils.openmax import compute_train_score_and_mavs_and_dists,fit_weibull,openmax,Evaluation

class OSREvaluator:
    def __init__(self,train_loader,visualizer,num_known_classes=7,exp_type="multi",processor="slotmax",use_softmax=False):
        super(OSREvaluator, self).__init__()
        self.best_val_loss=math.inf
        self.visualizer=visualizer
        self.use_softmax=use_softmax
        self.num_known_classes=num_known_classes
        self.closed_dataloader=train_loader
        self.exp_type=exp_type
        self.processor=processor
        if exp_type=="multi":
            self.slot_predictor=multi_correct_slot_2
            self.acc_metric="mAP"
            self.ood_detector = slot_max if self.processor!="openslot" else slot_score
        else:
            self.slot_predictor=get_correct_slot
            self.acc_metric="Acc"
            self.ood_detector = slot_max if self.processor!="openslot" else slot_score



    def postprocess(self, model, sample,compute_ood=True,phase="val"):
        slots,logits,__,matching_loss= model(sample)
        in_logits=self.slot_predictor(logits)
        if compute_ood:
            ood_score=self.ood_detector(logits,exp_type=self.exp_type,phase=phase)
        else:
            ood_score=torch.inf,
        return slots,in_logits,ood_score,matching_loss

    def get_known_unknown_scores(self,model,in_test_loader, out_test_loader_dict):
        _pred_k, _labels = [], []
        _val_ood_scores = []
        out_test_key_List = list(out_test_loader_dict.keys())
        _pred_u = {}
        with torch.no_grad():
            for batch_idx, sample in enumerate(in_test_loader):
                for key, values in sample.items():
                    sample[key] = values.cuda()
                slots, logits, ood_scores, matching_loss = self.postprocess(model, sample)

                _pred_k.append(logits.data.cpu().numpy())
                _val_ood_scores.append(ood_scores.data.cpu().numpy())
                _labels.append(sample["label"].data.cpu().numpy())

            for out_test_type in out_test_key_List:
                test_loader = out_test_loader_dict[out_test_type]
                _pred_u[out_test_type] = []
                for batch_idx, sample in enumerate(test_loader):
                    for key, values in sample.items():
                        sample[key] = values.cuda()
                    __, __, ood_scores, __ = self.postprocess(model, sample,phase="ood")
                    _pred_u[out_test_type].append(ood_scores.data.cpu().numpy())
                _pred_u[out_test_type] = np.concatenate(_pred_u[out_test_type], 0)
        _pred_k = np.concatenate(_pred_k, 0)
        _val_ood_scores = np.concatenate(_val_ood_scores, 0)

        return _pred_k,_labels,_pred_u,_val_ood_scores

    def eval(self, model,in_test_loader, out_test_loader_dict,epoch_idx,writer,compute_acc=True,osr=False):
        _pred_k,_labels=[],[]
        _val_ood_scores=[]
        out_test_key_List = list(out_test_loader_dict.keys())
        _pred_u={}
        with torch.no_grad():
            for batch_idx,sample in enumerate(in_test_loader):
                for key, values in sample.items():
                    sample[key] = values.cuda()
                slots,logits,ood_scores,matching_loss = self.postprocess(model, sample)

                _pred_k.append(logits.data.cpu().numpy())
                _val_ood_scores.append(ood_scores.data.cpu().numpy())
                _labels.append(sample["label"].data.cpu().numpy())
            self.slot_visualization(sample,model,writer,epoch_idx*batch_idx,"val")
            for out_test_type in out_test_key_List:
                test_loader=out_test_loader_dict[out_test_type]
                _pred_u[out_test_type]=[]
                for batch_idx, sample in enumerate(test_loader):
                    for key, values in sample.items():
                        sample[key] = values.cuda()
                    __, __,ood_scores, __ = self.postprocess(model, sample,phase="ood")
                    _pred_u[out_test_type].append(ood_scores.data.cpu().numpy())
                _pred_u[out_test_type]=np.concatenate(_pred_u[out_test_type], 0)
                self.slot_visualization(sample, model, writer, epoch_idx * batch_idx,"ood_"+out_test_type)
        _pred_k = np.concatenate(_pred_k, 0)
        _val_ood_scores= np.concatenate(_val_ood_scores, 0)
        if compute_acc:
            _labels = np.concatenate(_labels, 0)
            in_accuracy=self.compute_in_test_accuracy(_pred_k,_labels)
            writer.add_scalar("val/"+self.acc_metric+"",in_accuracy,epoch_idx)
            print("Val | Epoch: {:d}\t ".format(epoch_idx) + str(
                    self.acc_metric) + " (%): {:.3f}\t".format(in_accuracy))

        for key,u_predictions in _pred_u.items():
            print("start to evaluate "+key+" ood:")
            ood_evaluations=self.ood_eval(_val_ood_scores,u_predictions)
            writer.add_scalar(""+key+"/AUPR",ood_evaluations["AUPR"],epoch_idx)
            writer.add_scalar("" + key + "/AUROC", ood_evaluations["AUROC"],epoch_idx)
            writer.add_scalar("" + key + "/FPR@95", ood_evaluations["FPR@95"],epoch_idx)
            if osr:
                ood_evaluations["OSCR"]=evaluation.compute_oscr(_val_ood_scores,u_predictions,_labels)*100
                writer.add_scalar("" + key + "/OSCR",ood_evaluations["OSCR"] ,epoch_idx)
            print("Metrics",{
                "OOD": ood_evaluations
            })

    def ood_eval(self,k_predictions,u_predictions):

        if self.exp_type=="single":            # single-label OOD detction evaluation
            x1, x2 = np.max(k_predictions, axis=1), np.max(u_predictions, axis=1)
            results = evaluation.metric_ood(x1, x2)['Bas']
            ap_score = average_precision_score([0] * len(k_predictions) + [1] * len(u_predictions),
                                               list(-np.max(k_predictions, axis=-1)) + list(-np.max(u_predictions, axis=-1)))
            results['AUPR'] = ap_score * 100

        else:
            from utils.multi_ood_evaluator import get_multi_ood_results
            results=get_multi_ood_results(k_predictions, u_predictions)

        return results
    def compute_in_test_accuracy(self,known_pred,labels):
        if self.exp_type=="single":
            predictions = np.argmax(known_pred,axis=-1)
            correct = (predictions == labels.data).sum()
            acc_metric = correct / known_pred.shape[0]
        else:
            FinalMAPs = []
            for i in range(0, self.num_known_classes):
                precision, recall, thresholds = metrics.precision_recall_curve(labels[:,i], known_pred[:,i])

                FinalMAPs.append(metrics.auc(recall, precision))
            acc_metric=np.mean(FinalMAPs)
        acc_metric=acc_metric*100
        return acc_metric


    def openmax_processor(self,test_loader,out_test_loader_dict,model,epoch_idx,writer,compute_acc=True,osr=False):
        # Fit the weibull distribution from training data.
        #tail_best, alpha_best, th_best = None, None, None
        #auroc_best=0.
        print("Fittting Weibull distribution...")
        #for tailsize in [20, 40, 80]:
        _, mavs, dists = compute_train_score_and_mavs_and_dists(self.num_known_classes, self.closed_dataloader, model)
        categories = list(range(0, self.num_known_classes))
        weibull_model = fit_weibull(mavs, dists, categories, 20, "euclidean")

        _pred_k,_labels,_pred_u,_val_ood_scores=self.get_known_unknown_scores(model,test_loader, out_test_loader_dict)

        if compute_acc:
            _labels = np.concatenate(_labels, 0)
            in_accuracy=self.compute_in_test_accuracy(_pred_k,_labels)
            writer.add_scalar("val/"+self.acc_metric+"",in_accuracy,epoch_idx)
            print("Val | Epoch: {:d}\t ".format(epoch_idx) + str(
                    self.acc_metric) + " (%): {:.3f}\t".format(in_accuracy))
        _val_ood_scores=np.expand_dims(_val_ood_scores,axis=1)


            # for alpha in [3]:
            #     for th in [0.0, 0.5,0.6,0.65,0.7, 0.75, 0.8, 0.85, 0.9,0.95]:
        open_score_val = []
        val_predictions = []
        for idx,score in enumerate(_val_ood_scores):
            so=openmax(weibull_model, categories, score,
                             0.5, 3, "euclidean")

            open_score_val.append(so)
            val_predictions.append(np.argmax(so) if np.max(so) >= 0.90 else self.num_known_classes)
        open_score_val=np.array(open_score_val)
        ood_score_val = -1 * open_score_val[:, -1]


        #deter_metric=0.
        for key,u_predictions in _pred_u.items():
            open_score_test=[]
            open_score_pred=[]
            u_predictions=np.expand_dims(u_predictions,axis=1)
            #neg_label=[self.num_known_classes for i in range(u_predictions.shape[0])]
            print("start to evaluate "+key+" ood:")
            for idx, score in enumerate(u_predictions):
                so = openmax(weibull_model, categories, score,
                                               0.5, 3, "euclidean")

                open_score_test.append(so)

                open_score_pred.append(np.argmax(so) if np.max(so) >= 0.90 else self.num_known_classes)


            open_score_test = np.array(open_score_test)
            ood_score_test=-1 * open_score_test[:, -1]
            #x1, x2 = np.max(open_score_val, axis=1), np.max(open_score_test, axis=1)
            results = evaluation.metric_ood(ood_score_val, ood_score_test)['Bas']
            ap_score = average_precision_score([0] * len(ood_score_val) + [1] * len(ood_score_test),
                                               list(-ood_score_val) + list(-ood_score_test))
            results['AUPR'] = ap_score * 100
            #open_score_pred=np.array(open_score_pred)
            results['OSCR']=evaluation.compute_oscr(open_score_val, open_score_test, _labels,openmax=True)*100
            print(results)
            #deter_metric +=results['AUROC']

            # if deter_metric > auroc_best:
            #     tail_best, alpha_best, th_best = tailsize, alpha, th
            #     auroc_best = deter_metric

        # print("Best params:")
        # print(tail_best, alpha_best, th_best)

            # eval_openmax = Evaluation(open_pred, open_labels, open_scores)
            #
            # print(f"OpenMax accuracy is %.3f" % (eval_openmax.accuracy))
            # print(f"OpenMax F1 is %.3f" % (eval_openmax.f1_measure))
            # print(f"OpenMax f1_macro is %.3f" % (eval_openmax.f1_macro))
            # print(f"OpenMax f1_macro_weighted is %.3f" % (eval_openmax.f1_macro_weighted))
            # print(f"OpenMax area_under_roc is %.3f" % (eval_openmax.area_under_roc))
            # print(f"_________________________________________")

        # Get the prdict results.
        # scores = torch.cat(scores, dim=0).cpu().numpy()
        # labels = torch.cat(labels, dim=0).cpu().numpy()
        # scores = np.array(scores)[:, np.newaxis, :]
        # labels = np.array(labels)
        #
        #
        #
        # pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
        # score_softmax, score_openmax = [], []
        # for score in scores:
        #     so, ss = openmax(weibull_model, categories, score,
        #                      0.5, 3, "euclidean")  # openmax_prob, softmax_prob
        #     pred_softmax.append(np.argmax(ss))
        #     pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= 0.9 else self.num_known_classes)
        #     pred_openmax.append(np.argmax(so) if np.max(so) >= 0.9 else self.num_known_classes)
        #     score_softmax.append(ss)
        #     score_openmax.append(so)
        #
        # print("Evaluation...")
        # eval_softmax = Evaluation(pred_softmax, labels, score_softmax)
        # eval_softmax_threshold = Evaluation(pred_softmax_threshold, labels, score_softmax)

        #
        # print(f"SoftmaxThreshold accuracy is %.3f" % (eval_softmax_threshold.accuracy))
        # print(f"SoftmaxThreshold F1 is %.3f" % (eval_softmax_threshold.f1_measure))
        # print(f"SoftmaxThreshold f1_macro is %.3f" % (eval_softmax_threshold.f1_macro))
        # print(f"SoftmaxThreshold f1_macro_weighted is %.3f" % (eval_softmax_threshold.f1_macro_weighted))
        # print(f"SoftmaxThreshold area_under_roc is %.3f" % (eval_softmax_threshold.area_under_roc))
        # print(f"_________________________________________")
        #

    def slot_visualization(self,sample,model,writer,step,phase):
        image = sample['img'].cuda()
        decoder_output = model.get_slot_attention_mask(image)
        log_visualizations(self.visualizer, writer, decoder_output, image, step,phase=phase)
