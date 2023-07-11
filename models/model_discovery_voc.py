import torch
import torch.nn.functional as F
import models.vision_transformer as vits
import models.resnet_s as resnet
import scipy
import torch.nn as nn
from models.slotattention import SlotAttention
from torchsummary import summary
import torchvision
from scipy.optimize import linear_sum_assignment
import numpy as np
from itertools import chain
from utils.utils import set_trainable,initialize_weights
import timm
arch_config=\
{
    'resnet': {'feats_channel':2048},
    "vit_small": {'feats_channel':384},
    "vit_base": {'feats_channel': 768}
}

class PositionEmbeddingLearned(nn.Module):
    # https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    def __init__(self, resolution,num_pos_feats=64):
        super().__init__()
        self.row_embed = nn.Embedding(resolution[0], num_pos_feats//2)
        self.col_embed = nn.Embedding(resolution[1], num_pos_feats//2)
        #TODO: assert that x.shape matches the passed row_embed, col_embed
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        #print(x.shape)
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)

        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return x + pos




class MLP_decoder(nn.Module):
    def __init__(self,num_mlp=4,in_channel=64,resolution=(32,32),hid_dim=1024,out_channel=784):
        super().__init__()
        self.decoder_initial_size = resolution
        self.decoder_pos = PositionEmbeddingLearned(resolution,num_pos_feats=in_channel)
        self.decoder=[]
        self.decoder.append(nn.Linear(in_channel,hid_dim))
        self.feature_channel=out_channel
        for i in range(1,num_mlp-1):
            self.decoder.append(nn.Linear(hid_dim,hid_dim))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(hid_dim,out_channel+1))
        self.decoder=nn.Sequential(*self.decoder)

    def forward(self,x):
        x=self.decoder_pos(x)
        batch,num_slots, height,width=x.shape
        x=x.reshape(batch,num_slots,-1).permute(0,2,1)
        x=self.decoder(x)

        x=x.reshape(batch,self.feature_channel+1,height,width)

        return x




class Net(nn.Module):
    def __init__(self,arch='vit_base',patch_size=16,image_size=224,resume='./checkpoints/dino_vitbase16_pretrain.pth',num_mlp_decoder=4,num_slot=6,slot_dim=256,mlp_hidden_size=64,num_iteration=3,freeze_encoder=True):
        super().__init__()
        ## initialize the feature encoder
        self.arch=arch
        if 'vit' in self.arch:
            self.encoder =vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            self.feats_channels=arch_config[self.arch]['feats_channel']
            resolution=(image_size//patch_size,image_size//patch_size)
            self.feats_resolution=resolution
        else:
            from torchvision.models.resnet import resnet50
            self.encoder  = resnet50()
            self.feats_channels=arch_config['resnet']['feats_channel']
            self.feats_resolution=(image_size[0]//32,image_size[1]//32)
        state_dict=torch.load(resume)
        msg=self.encoder.load_state_dict(state_dict,strict=False)
        if 'resnet' in self.arch:
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            self.encoder=torch.nn.Sequential(*list(self.encoder.children())[:-1])
        self.embidding_layer=nn.Linear(self.feats_channels, self.feats_channels) ## transform the encoder feature before slot attention
        self.num_slots=num_slot
        self.num_iterations=num_iteration
        self.slot_attention = SlotAttention(
            num_iter=self.num_iterations,
            num_slots=self.num_slots,
            input_size=self.feats_channels,
            slot_size=slot_dim,
            mlp_hidden_size=mlp_hidden_size)

        self.decoder=MLP_decoder(num_mlp=num_mlp_decoder,in_channel=slot_dim,resolution=self.feats_resolution,
                                 hid_dim=1024,out_channel=self.feats_channels)
        if freeze_encoder:
            set_trainable([self.encoder], False)

        self.criterion=nn.MSELoss()

    def encoder_feature_extractor(self,x):
        if self.arch=='resnet':
            x= self.encoder(x)
            batch,dim,height,width=x.shape
            x=x.reshape(batch,dim,-1)
            x=x.permute(0,2,1)
        else:
            x= self.encoder.forward_feats(x)
        return x


    def forward(self, image):
        feature_supervision=self.encoder_feature_extractor(image)
        target_feature=feature_supervision.clone().detach()
        #print(target_feature.shape)
        embedding_feature=self.embidding_layer(feature_supervision)
        #Slot Attention module.
        slots, __, __= self.slot_attention(embedding_feature)
        batch_size, num_slots, slot_size=slots.shape
        slots=slots.reshape((-1, slots.shape[-1]))

        slots = slots.unsqueeze(1).unsqueeze(2)

        slots = slots.repeat((1, self.feats_resolution[0], self.feats_resolution[1], 1)).permute(0,3,1,2).contiguous()
        ## decoding
        slots=self.decoder(slots)
        soft_mask=slots[:,-1].reshape(batch_size,num_slots,self.feats_resolution[0], self.feats_resolution[1])
        soft_mask=F.softmax(soft_mask,1).reshape(batch_size,num_slots,-1)
        feats_recons=slots[:,:-1].reshape(batch_size,num_slots,self.feats_channels,-1)
        feats_recons=torch.einsum('bsr,bsdr->bdr',soft_mask,feats_recons).permute(0,2,1)
        feats_recons_loss=self.criterion(feats_recons,target_feature)
        soft_mask=soft_mask.reshape(batch_size,num_slots,self.feats_resolution[0], self.feats_resolution[1]).contiguous()


        return slots,soft_mask,feats_recons_loss


    def get_encoder_params(self):
        return self.encoder.parameters()

    def get_slotattention_params(self):
        return self.slot_attention.parameters()

    def get_decoder_params(self):
        return chain(self.slot_attention.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__=="__main__":
    #print(timm.list_models(pretrained=True))
    #print(timm.create_model('vit_base_patch16_224',checkpoint_path='../checkpoints/dino_vitbase16_pretrain.pth'))
    model=Net(image_size=(256,256)).cuda()
    model.eval()
    input=torch.rand((2,3,256,256)).cuda()
    slots,slots_mask,feat_loss=model(input)
    print(slots_mask.shape)

