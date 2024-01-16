import torch
from torch.autograd import Variable
import numpy as np





def sample_estimator(model,num_classes,closed_loader,slot_dim=256):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []

    for j in range(num_classes):
        list_features.append(0)

    with torch.no_grad():
        for sample in closed_loader:
            for key, values in sample.items():
                sample[key] = values.cuda()
            outputs = model(sample)
            labels=sample['class_label']
            slot_features,logits,indices= outputs["slots"],outputs["fg_pred"],outputs['fg_indices']##[batch, num_slot,dim]

            for  idx,(feature, t,label,indice) in enumerate(zip(slot_features,logits, labels,indices)):

                cls_indices=torch.nonzero(label)[:,1]
                for i in range(indice.shape[-1]):
                    if indice[0,i]<len(cls_indices): # check if the indice stand for a valid class
                        determined_logit=t[indice[1,i]]
                        target_cls=cls_indices[indice[0,i]]
                        if torch.argmax(determined_logit) == target_cls: # correctly predicted slors
                            slot_fea=feature[indice[1,i]].unsqueeze(0)
                            if num_sample_per_class[target_cls] == 0:
                                list_features[target_cls] = slot_fea
                            else:
                                list_features[target_cls] = torch.concatenate([list_features[target_cls],
                                                                  slot_fea], 0)
                            num_sample_per_class[target_cls] += 1

    num_feature = slot_dim
    temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
    for j in range(num_classes):
        temp_list[j] = torch.mean(list_features[j], 0)
    sample_class_mean = temp_list
    X = 0
    for i in range(num_classes):
        if i == 0:
            X = list_features[i] - sample_class_mean[i]
        else:
            X = torch.cat((X, list_features[i] - sample_class_mean[i]), 0)
    # find inverse
    group_lasso.fit(X.cpu().numpy())
    temp_precision = group_lasso.precision_
    temp_precision = torch.from_numpy(temp_precision).float().cuda()
    precision = temp_precision

    return sample_class_mean, precision