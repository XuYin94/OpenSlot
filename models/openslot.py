import torch
import torch.nn.functional as F
import models.vision_transformer as vits
import models.resnet_s as resnet
import scipy
import hydra_zen
import torch.nn as nn
from typing import Any, Dict, Optional
import torchvision
from scipy.optimize import linear_sum_assignment
import numpy as np
from itertools import chain
from utils.utils import set_trainable
import timm
import yaml
from omegaconf import DictConfig, OmegaConf
from ocl.utils.routing import Combined

class Net(nn.Module):
    def __init__(self,model_config,checkpoint_path="D:\\Open_world_recognition_with_object_centric_learning\\saved_models\\Discovery\\osr_voc_transformer_decoder\\model_final.ckpt"):
        super().__init__()
        model = hydra_zen.instantiate(model_config, _convert_="all")
        # print(models)
        self.feature_extractor=model["feature_extractor"]
        self.conditioning=model["conditioning"]
        self.perceptual_grouping=model['perceptual_grouping']
        self.osr_classifier=model['classifier']
        self.decoder=model['object_decoder']
        if checkpoint_path is not None:
            self.__load_discovery_weights(checkpoint_path)

    def __load_discovery_weights(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)["state_dict"]
        for key in list(checkpoint.keys()):
            if 'models.' in key:
                checkpoint[key.replace('models.', '')] = checkpoint[key]
                del checkpoint[key]
        module_dict=dict( ('.'.join(key.split('.')[1:]), value) for (key, value) in checkpoint.items() if 'feature_extractor' in key)
        self.feature_extractor.load_state_dict(module_dict)
        module_dict=dict( ('.'.join(key.split('.')[1:]), value) for (key, value) in checkpoint.items() if 'perceptual_grouping' in key)
        self.perceptual_grouping.load_state_dict(module_dict)
        module_dict=dict( ('.'.join(key.split('.')[1:]), value) for (key, value) in checkpoint.items() if 'conditioning' in key)
        self.conditioning.load_state_dict(module_dict)
        module_dict=dict( ('.'.join(key.split('.')[1:]), value) for (key, value) in checkpoint.items() if 'decoder' in key)
        self.decoder.load_state_dict(module_dict)

    def forward(self, images,target=None):
        batch_size = images.shape[0]

        features = self.feature_extractor(video=images)
        conditioning = self.conditioning(batch_size=batch_size)

        perceptual_grouping_output = self.perceptual_grouping(
            feature=features, conditioning=conditioning
        )
        slots = perceptual_grouping_output.objects
        pred=self.osr_classifier(slots)
        if target is not None:
            matching_loss,__=self.__hungarianMatching(pred,target)
            return slots,pred,matching_loss
        else:
            return slots,pred


        return pred

    def get_slot_attention_mask(self,images):
        batch_size = images.shape[0]
        features = self.feature_extractor(video=images)
        target=features.features
        conditioning = self.conditioning(batch_size=batch_size)
        perceptual_grouping_output = self.perceptual_grouping(
            feature=features, conditioning=conditioning
        )
        object_features = perceptual_grouping_output.objects
        feature_attributes=perceptual_grouping_output.feature_attributions

        # print(type(feature_attributes))
        # print(type(object_features))
        # print(type(target))

        output=self.decoder(object_features,feature_attributes,target,images)
        return output


    def get_trainable_params_groups(self,different_lr=True):
        set_trainable([self.feature_extractor,self.decoder], False) ## freeze the feature extractor

        if different_lr:
            discovery_params=chain(self.conditioning.parameters(), self.perceptual_grouping.parameters())
            trainable_params = [{'params': filter(lambda p: p.requires_grad, discovery_params)},
                                {'params': filter(lambda p: p.requires_grad, self.osr_classifier.parameters())}]
        else:
            set_trainable([self.conditioning,self.perceptual_grouping], False)
            trainable_params=filter(lambda p: p.requires_grad, self.osr_classifier.parameters())
        return trainable_params

    def __hungarianMatching(self,class_pred, targets):

        """
        match the slot-level prediction with the ground truth
        "class_pred: [batch,num_slot,num_classes]"
        "targets: [batch,max_num_object, num_classes], after one-hot encoding"
        """
        class_pred=class_pred.unsqueeze(1) ##[batch,1, num_slot, num_classes]
        targets=targets.unsqueeze(2) ##[batch,max_num_object,1,num_classes]
        cost_matrix=(-(targets * class_pred.log_softmax(dim=-1))).sum(dim=-1)
        indices = np.array(
            list(map(scipy.optimize.linear_sum_assignment, cost_matrix.cpu().detach().numpy())))

        indices = torch.LongTensor(np.array(indices)) ##[batch, 2,min(max_num_object,num_slot)]

        smallest_cost_matrix = torch.stack(
            [
                cost_matrix[i][indices[i, 0], indices[i, 1]]
                for i in range(cost_matrix.shape[0])
            ]
        )
        device=cost_matrix.device
        return smallest_cost_matrix[:,0].sum().to(device),indices.to(device)


if __name__=="__main__":
    #print(timm.list_models(pretrained=True))
    #print(timm.create_model('vit_base_patch16_224',checkpoint_path='../checkpoints/dino_vitbase16_pretrain.pth'))
    model=Net(model_config="../configs/classification_config.yaml").cuda()
    model.eval()
    input=torch.rand((2,3,224,224)).cuda()
    params=model.get_trainable_params_groups(different_lr=True)

    #print(slots_mask.shape)

