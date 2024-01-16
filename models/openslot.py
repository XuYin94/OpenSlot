import torch
import torch.nn.functional as F
import models.vision_transformer as vits
import models.resnet_s as resnet
import scipy
import hydra_zen
import math
import cv2
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
    def __init__(self, model_config, checkpoint_path="./checkpoints/model_final.ckpt"):
        super().__init__()
        model = hydra_zen.instantiate(model_config, _convert_="all")
        # print(models)
        self.feature_extractor = model["feature_extractor"]
        self.conditioning = model["conditioning"]
        self.perceptual_grouping = model['perceptual_grouping']
        self.osr_classifier = model['classifier']
        self.aux_classifier = model['aux_classifier']
        self.decoder = model['object_decoder']
        self.attention_threshold=model['attention_threshold']
        self.noisy_threshold=model['noise_threshold']
        if checkpoint_path is not None:
            self.__load_discovery_weights(checkpoint_path)

    def __load_discovery_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)["state_dict"]
        for key in list(checkpoint.keys()):
            if 'models.' in key:
                checkpoint[key.replace('models.', '')] = checkpoint[key]
                del checkpoint[key]
        module_dict = dict(
            ('.'.join(key.split('.')[1:]), value) for (key, value) in checkpoint.items() if 'feature_extractor' in key)
        msg = self.feature_extractor.load_state_dict(module_dict)
        # print(msg)
        module_dict = dict(('.'.join(key.split('.')[1:]), value) for (key, value) in checkpoint.items() if
                           'perceptual_grouping' in key)
        msg = self.perceptual_grouping.load_state_dict(module_dict)
        # print(msg)
        module_dict = dict(
            ('.'.join(key.split('.')[1:]), value) for (key, value) in checkpoint.items() if 'conditioning' in key)
        msg = self.conditioning.load_state_dict(module_dict)
        # print(msg)
        module_dict = dict(
            ('.'.join(key.split('.')[1:]), value) for (key, value) in checkpoint.items() if 'decoder' in key)
        msg = self.decoder.load_state_dict(module_dict)
        print("Successfully load the discovery weight !!")

    def forward(self, sample,using_bg_pred=True):
        #print(self.attention_threshold)
        output = {}
        images = sample["img"]
        batch_size = images.shape[0]

        features = self.feature_extractor(video=images)
        conditioning = self.conditioning(batch_size=batch_size)

        perceptual_grouping_output = self.perceptual_grouping(
            feature=features, conditioning=conditioning
        )
        slots = perceptual_grouping_output.objects
        pred = self.osr_classifier(slots)
        bg_pred = self.aux_classifier(slots)
        output["slots"] = slots
        output["fg_pred"] = pred
        output["bg_pred"] = bg_pred

        feature_attributes = perceptual_grouping_output.feature_attributions
        target = features.features
        masks = self.decoder(slots.detach(), feature_attributes, target,
                                images).masks  ## get the attention mask of every slot
        valid_slots=self.get_valid_slots(masks)
        output["valid_slots"] = valid_slots
        if 'class_label' in sample:
            fg_class_label = sample['class_label'].cuda()
            fg_slot_selection = sample['fg_channel'].cuda()
            if not using_bg_pred:
                output["fg_matching_loss"], output["fg_indices"] = self.__loss_matcher(pred, fg_class_label,
                                                                                    fg_slot_selection,
                                                                                    valid_semantic_slots=valid_slots)
            else:
                output["fg_matching_loss"], output["fg_indices"] = self.__loss_matcher(pred, fg_class_label,
                                                                    fg_slot_selection, valid_semantic_slots=valid_slots,aux_pred=bg_pred)
                          
            fg_indices = output["fg_indices"]
            background_label = torch.ones_like(bg_pred).float()  ##[batch,num_slot,1]
            for i in range(fg_indices.shape[0]):
                valid_indices = fg_indices[i, 1, fg_slot_selection[i, :, 0] > 0]
                assert fg_slot_selection[i, :, 0].sum() == valid_indices.shape[0]
                background_label[i, valid_indices] = 0.
            background_label[valid_slots == 0] = 1.0
            output["bg_loss"] = F.binary_cross_entropy_with_logits(bg_pred, background_label)

        return output

    def get_slot_attention_mask(self, images, sorted=True, using_softmax=False):
        batch_size = images.shape[0]
        features = self.feature_extractor(video=images)
        # __,__,width,height=features.shape
        target = features.features
        conditioning = self.conditioning(batch_size=batch_size)
        perceptual_grouping_output = self.perceptual_grouping(
            feature=features, conditioning=conditioning
        )
        object_features = perceptual_grouping_output.objects  ## slots: [batch, num_slots, slot_dim]
        feature_attributes = perceptual_grouping_output.feature_attributions
        if sorted:
            logits = self.aux_classifier(object_features)  ##logits: [batch,num_slots,num_classes]
            # print(logits.shape)
            if using_softmax:
                logits = torch.softmax(logits, dim=-1)
            slot_maximum, __ = torch.max(logits, dim=-1)  # select the largest prediction
            # print(slot_maximum)
            __, slot_indices = torch.sort(slot_maximum, dim=-1)
            sorted_slots = [object_features[i, slot_indices[i]] for i in range(batch_size)]
            sorted_slots = torch.stack(sorted_slots, dim=0)
            output = self.decoder(sorted_slots, feature_attributes, target, images)
        else:
            output = self.decoder(object_features, feature_attributes, target, images)

        masks = output.masks
        valid_slots=self.get_valid_slots(masks)
        output.masks_as_image[valid_slots == 0] = 1.


        return output

    def get_trainable_params_groups(self, different_lr=False):
        set_trainable([self.feature_extractor, self.decoder], False)  ## freeze the feature extractor

        if different_lr:
            discovery_params = chain(self.conditioning.parameters(), self.perceptual_grouping.parameters())
            classifier_params = chain(self.osr_classifier.parameters(), self.aux_classifier.parameters())
            trainable_params = [{'params': filter(lambda p: p.requires_grad, discovery_params)},
                                {'params': filter(lambda p: p.requires_grad, classifier_params)}]
        else:
            #set_trainable([self.conditioning, self.perceptual_grouping,self.aux_classifier], False)
            classifier_params = chain(self.osr_classifier.parameters(),self.aux_classifier.parameters())
            trainable_params = [{'params': filter(lambda p: p.requires_grad, classifier_params)}]

        return trainable_params

    def __loss_matcher(self, class_pred, targets, valid_category, valid_semantic_slots=None, aux_pred=None):

        """
        match the slot-level prediction with the ground truth
        "class_pred: [batch,num_slot,num_classes]"
        "targets: [batch,max_num_object, num_classes], after one-hot encoding"
        Returns:
            indices:
                The first column the indices of the true categories while the second
                column is the the indices of the slots.
        """
        class_pred = class_pred.unsqueeze(1)  ##[batch,1, num_slot, num_classes]
        targets = targets.unsqueeze(2)
        device = class_pred.device

        # if class_pred.shape[-1] == 1:
        #     cost_matrix = (-(targets * nn.LogSigmoid()(class_pred))).sum(dim=-1)
        #     weight_matrix = cost_matrix
        # else:
        valid_semantic_slots = valid_semantic_slots.unsqueeze(1)
        cost_matrix = (-(targets * class_pred.log_softmax(dim=-1))).sum(dim=-1)


        if aux_pred is not None:  ## assign the high-confidence BG slot and invalid slots  with high cost.
            #bg_pred=torch.sigmoid(aux_pred)
            bg_minimum=torch.min(aux_pred,dim=1,keepdim=True)[0]
            bg_maxmim=torch.max(aux_pred,dim=1,keepdim=True)[0]
            bg_pred=(aux_pred-bg_minimum)/(bg_maxmim-bg_minimum+1e-10)## min-max normalization
            bg_pred=bg_pred.permute(0,2,1)
            weight_matrix = cost_matrix + 50000 * ((bg_pred > self.noisy_threshold)|(1-valid_semantic_slots))
        else:
            weight_matrix = (cost_matrix + 50000 * (1 - valid_semantic_slots))
        weight_matrix = weight_matrix * valid_category + 100000 * (1 - valid_category)

        __, indices = self.__hungarianMatching(weight_matrix)
        batch_range = torch.arange(cost_matrix.shape[0]).unsqueeze(-1)
        loss_per_object = cost_matrix[batch_range, indices[:, 0], indices[:, 1]]
        #print(valid_category.shape)
         
        return (loss_per_object.sum()/valid_category.sum()).to(device), indices

    def __hungarianMatching(self, weight_matrix):
        indices = np.array(
            list(map(scipy.optimize.linear_sum_assignment, weight_matrix.cpu().detach().numpy())))

        indices = torch.LongTensor(np.array(indices))  ##[batch, 2,min(max_num_object,num_slot)]
        smallest_cost_matrix = torch.stack(
            [
                weight_matrix[i][indices[i, 0], indices[i, 1]]
                for i in range(weight_matrix.shape[0])
            ]
        )
        device = weight_matrix.device
        return smallest_cost_matrix.to(device), indices.to(device)

    def get_valid_slots(self,masks):
        batch_size,num_slots=masks.shape[:2]
        device=masks.device
        valid_slots=torch.ones((batch_size,num_slots))
        for i in range(batch_size):
            for j in range(num_slots):
                single_mask=(masks[i,j]>self.attention_threshold).cpu().numpy().astype(np.uint8)
                if single_mask.sum()>=2:
                    __,__,stats, __ = cv2.connectedComponentsWithStats(single_mask, connectivity=8)
                    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                    largest_component_area = stats[largest_component_label, cv2.CC_STAT_AREA]

                    valid_slots[i,j]=largest_component_area

        return (valid_slots>=2).long().to(device)
    
    def forward_slots(self,images):
        batch_size = images.shape[0]

        features = self.feature_extractor(video=images)
        conditioning = self.conditioning(batch_size=batch_size)

        perceptual_grouping_output = self.perceptual_grouping(
            feature=features, conditioning=conditioning
        )
        slots = perceptual_grouping_output.objects

        return slots
if __name__ == "__main__":
    # print(timm.list_models(pretrained=True))
    # print(timm.create_model('vit_base_patch16_224',checkpoint_path='../checkpoints/dino_vitbase16_pretrain.pth'))
    model = Net(model_config="../configs/single_voc_classification_config.yaml").cuda()
    model.eval()
    input = torch.rand((2, 3, 224, 224)).cuda()
    params = model.get_trainable_params_groups(different_lr=True)

    # print(slots_mask.shape)

