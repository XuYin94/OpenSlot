import torch
import numpy as np
from sklearn import metrics
import tqdm
import math
from utils.utils import get_highest_slot,multi_correct_slot,slot_max,slot_min,slot_score,log_visualizations
from utils import evaluation
from sklearn.metrics import average_precision_score
from utils.openmax import compute_train_score_and_mavs_and_dists,fit_weibull,openmax

get_ood_detector={
    'slotmax':slot_max,
    'slotmin':slot_min,
    'openslot':slot_score
}
to_np = lambda x: x.data.cpu().numpy()

class OSREvaluator:
    def __init__(self,train_loader,visualizer,num_known_classes=7,exp_type="multi",processor="slotmin",use_softmax=False):
        super(OSREvaluator, self).__init__()
        self.best_val_loss=math.inf
        self.visualizer=visualizer
        self.use_softmax=use_softmax
        self.num_known_classes=num_known_classes
        self.closed_dataloader=train_loader
        self.exp_type=exp_type
        self.processor=processor
        self.ood_detector = get_ood_detector[self.processor]
        if exp_type=="multi":
            self.slot_predictor=multi_correct_slot
            self.acc_metric="mAP"
        else:
            self.slot_predictor=get_highest_slot
            self.acc_metric="Acc"



    def postprocess(self, model, sample,compute_ood=True,phase="val"):
        slots,logits,__,matching_loss= model(sample)
        in_logits=self.slot_predictor(logits)
        if compute_ood:
            ood_score=self.ood_detector(logits,exp_type=self.exp_type,phase=phase)
        else:
            ood_score=torch.inf,
        return slots,in_logits,ood_score,matching_loss

    def get_known_unknown_scores(self,model,in_test_loader, out_test_loader_dict,writer,epoch_idx):
        _k_logits, _labels = [], []
        out_test_key_List = list(out_test_loader_dict.keys())
        _u_logits = {}
        with torch.no_grad():
            for batch_idx, sample in enumerate(in_test_loader):
                for key, values in sample.items():
                    sample[key] = values.cuda()
                slots, logits, __, __ = model(sample)

                _k_logits.append(logits.data.cpu())
                _labels.append(sample["label"].data.cpu())
            self.slot_visualization(sample,model,writer,epoch_idx*batch_idx,"val")
            for out_test_type in out_test_key_List:
                test_loader = out_test_loader_dict[out_test_type]
                _u_logits[out_test_type] = []
                for batch_idx, sample in enumerate(test_loader):
                    for key, values in sample.items():
                        sample[key] = values.cuda()
                    __,logits,__ , __ = model(sample)
                    _u_logits[out_test_type].append(logits.data.cpu())
                _u_logits[out_test_type] = torch.concatenate(_u_logits[out_test_type], 0)
                self.slot_visualization(sample, model, writer, epoch_idx * batch_idx,"ood_"+out_test_type)

        _k_logits = torch.concatenate(_k_logits, 0)
        _labels = torch.concatenate(_labels, 0)
        return _k_logits,_labels,_u_logits

    def eval(self, model,in_test_loader, out_test_loader_dict,epoch_idx,writer,compute_acc=True,osr=False):
        _known_logits,_labels,_unknown_logits=self.get_known_unknown_scores(model,in_test_loader,out_test_loader_dict,writer,epoch_idx)
        _correct_slot_logits = to_np(self.slot_predictor(_known_logits))
        if compute_acc:
            _labels = to_np(_labels)
            in_accuracy=self.compute_in_test_accuracy(_correct_slot_logits,_labels)
            writer.add_scalar("val/"+self.acc_metric+"",in_accuracy,epoch_idx)
            print("Val | Epoch: {:d}\t ".format(epoch_idx) + str(
                    self.acc_metric) + " (%): {:.3f}\t".format(in_accuracy))
        _known_ood_score=self.ood_detector(_known_logits,self.exp_type)
        for key,unknown_logit in _unknown_logits.items():
            print("start to evaluate "+key+" ood:")
            _unknown_ood_score=self.ood_detector(unknown_logit,self.exp_type)
            ood_evaluations=self.ood_eval(_known_ood_score,_unknown_ood_score)
            writer.add_scalar(""+key+"/AUPR",ood_evaluations["AUPR"],epoch_idx)
            writer.add_scalar("" + key + "/AUROC", ood_evaluations["AUROC"],epoch_idx)
            writer.add_scalar("" + key + "/FPR@95", ood_evaluations["FPR@95"],epoch_idx)
            if osr:
                _unknown_slot_logits = to_np(self.slot_predictor(unknown_logit))
                ood_evaluations["OSCR"]=evaluation.compute_oscr(_correct_slot_logits,_unknown_slot_logits,_labels)*100
                writer.add_scalar("" + key + "/OSCR",ood_evaluations["OSCR"] ,epoch_idx)
            print("Metrics",{
                "OOD": ood_evaluations
            })

    def ood_eval(self,k_ood_score,u_ood_score):
        #if self.exp_type=="single":
        results = evaluation.metric_ood(k_ood_score, u_ood_score)['Bas']
        ap_score = average_precision_score([0] * len(k_ood_score) + [1] * len(u_ood_score),
                                           list(-k_ood_score) + list(-u_ood_score))
        results['AUPR'] = ap_score*100
        # else:
        #     from utils.multi_ood_evaluator import get_multi_ood_results
        #     results = get_multi_ood_results(k_ood_score, u_ood_score)


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

        known_logits,_labels,unknown_logits=self.get_known_unknown_scores(model,test_loader, out_test_loader_dict,writer,epoch_idx)
        _correct_slot_logits = to_np(self.slot_predictor(known_logits))
        if compute_acc:
            _labels = to_np(_labels)
            in_accuracy=self.compute_in_test_accuracy(_correct_slot_logits,_labels)
            writer.add_scalar("val/"+self.acc_metric+"",in_accuracy,epoch_idx)
            print("Val | Epoch: {:d}\t ".format(epoch_idx) + str(
                    self.acc_metric) + " (%): {:.3f}\t".format(in_accuracy))


        if self.exp_type=="single":
            known_logits = np.expand_dims(_correct_slot_logits,1)
        known_logits=np.expand_dims(known_logits,axis=2)
        batch,nbr_slot,__,nbr_cls=known_logits.shape
        open_score_val = np.zeros((batch,nbr_slot,nbr_cls+1))
        print(open_score_val.shape)
        for idx,scores in enumerate(known_logits):
            for i,score in enumerate(scores):
                print(score.shape)
                if np.max(np.exp(score)/np.exp(score).sum())>0.95:
                    so=openmax(weibull_model, categories, score,
                                     0.5, 3, "euclidean")

                    open_score_val[idx,i]=so
        ood_score_val = -1 * np.max(open_score_val[:,:, -1],axis=-1)



        for key,u_predictions in unknown_logits.items():
            if self.exp_type == "single":
                u_predictions = np.expand_dims(to_np(self.slot_predictor(u_predictions)),axis=1)
            open_score_test=np.zeros((u_predictions.shape[0],u_predictions.shape[1],nbr_cls+1))
            u_predictions=np.expand_dims(u_predictions,axis=2)
            #neg_label=[self.num_known_classes for i in range(u_predictions.shape[0])]
            print("start to evaluate "+key+" ood:")

            for idx, scores in enumerate(u_predictions):
                for i, score in enumerate(scores):
                    if np.max(np.exp(score) / np.exp(score).sum()) > 0.95:
                        so = openmax(weibull_model, categories, score,
                                     0.5, 3, "euclidean")

                        open_score_test[idx,i]=so

            # for idx, score in enumerate(u_predictions):
            #     so = openmax(weibull_model, categories, score,
            #                                    0.5, 3, "euclidean")
            #
            #     open_score_test.append(so)
            #
            #     open_score_pred.append(np.argmax(so) if np.max(so) >= 0.90 else self.num_known_classes)
            #
            #
            # open_score_test = np.array(open_score_test)
            ood_score_test=-1 * np.max(open_score_test[:,:, -1],axis=-1)
            #x1, x2 = np.max(open_score_val, axis=1), np.max(open_score_test, axis=1)
            results = evaluation.metric_ood(ood_score_val, ood_score_test)['Bas']
            ap_score = average_precision_score([0] * len(ood_score_val) + [1] * len(ood_score_test),
                                               list(-ood_score_val) + list(-ood_score_test))
            results['AUPR'] = ap_score * 100
            #open_score_pred=np.array(open_score_pred)
            #results['OSCR']=evaluation.compute_oscr(open_score_val, open_score_test, _labels,openmax=True)*100
            print(results)


    def slot_visualization(self,sample,model,writer,step,phase):
        image = sample['img'].cuda()
        decoder_output = model.get_slot_attention_mask(image)
        log_visualizations(self.visualizer, writer, decoder_output, image, step,phase=phase)
