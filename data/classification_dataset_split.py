from torch.utils.data import Dataset,DataLoader
import os
import torch
import numpy as np
from PIL import Image
import random
from scipy.io import loadmat
import json
from torchvision import transforms
from torchvision.utils import make_grid

voc_cls_labels_dict = np.load('../data/cls_labels_voc.npy', allow_pickle=True).item()
def create_VOC_osr_class_split(name="voc",root_path="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\"):
    print("Creating {0} OSR split".format(name))
    name_list = {}
    mixed_name_list=[]
    cls_label_dict=voc_cls_labels_dict

    datafile = loadmat(os.path.join(root_path,"voc12-train.mat"))
    GT = datafile['labels']
    Imglist = datafile['Imlist']
    nbr_cls=np.zeros(20)
    for i in range(20):
        name_list[str(i)]=[]
    nbr_known_cls=5
    nbr_img_per_cls=200


    for idx,img in enumerate(Imglist):
        label=GT[idx]
        key=str(img).split('/')[-1][:-4]
        print(cls_label_dict[int(key)])
        if label.sum()==1:
            nbr_cls[label==1]+=1
            name_list[str((label==1).nonzero()[0][0])].append(img)
        else:
            mixed_name_list.append(img)
    full_cls_set=list((nbr_cls>=50).nonzero()[0]) ## number of the object categories that size over 50
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])## number of the object categories that size over 450
    print(full_train_cls)
    print(full_cls_set)

if __name__=="__main__":
    create_VOC_osr_class_split()