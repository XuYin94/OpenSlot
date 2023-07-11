import os
import os.path
import cv2
import numpy as np
from utils import transform
from torch.utils.data import Dataset,DataLoader
from utils import colorization
import PIL
from PIL import Image
from torchvision.utils import make_grid
from torchvision import transforms
import torch


class VOC_Dataset(Dataset):
    def __init__(self, split='train_aug', data_root=None,  transform=None):
        self.split = split
        self.data_root=data_root
        self.nbr_classes=21
        self.data_list=self.__make_list(data_root,split)
        self.transform = transform  ## Operations for data augmentation
        self.max_num_object=6
        #self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        #self.STD = [0.23965294, 0.23532275, 0.2398498]
        self.palette=colorization.get_voc_palette(21)
    def __make_list(self,data_root,split):

        assert split in ['train_aug','val_aug','test']
        name_list=[]
        text_file=os.path.join(data_root,"ImageSets/Segmentation",split+'.txt')
        line_list=open(text_file).readlines()
        print("Totally have {} samples in {} set.".format(len(line_list),str(split)))
        for line in line_list:
            line = line.strip()
            line_split = line.split(' ')
            input_name = self.data_root+line_split[0]
            if split in ['train_aug','val_aug']:
                label_name = self.data_root+line_split[1]
                salience_name=os.path.join(self.data_root,"salience",line_split[0].split("/")[2][:-3]+"png")
            else:
                label_name=input_name
                salience_name=input_name
            name_item=(input_name,label_name,salience_name)
            name_list.append(name_item)
        return name_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        input_path,label_path,salience_path=self.data_list[index]
        input=Image.open(input_path)
        label=Image.open(label_path)

        salience_map=Image.open(salience_path).convert("L")
        if self.transform is not None:
            sample=self.transform({'img':input,'label':label,'salience':salience_map})

        semantic_label=torch.unique(sample['label'])
        semantic_label=semantic_label[(semantic_label!=255)&(semantic_label!=0)]
        class_label=torch.zeros((self.max_num_object))
        num_class=semantic_label.shape[0]
        assert semantic_label.shape[0]<=self.max_num_object
        class_label[:num_class]=semantic_label
        sample['fg_channel']=class_label>0
        class_label=torch.as_tensor(class_label,dtype=torch.int64)
        class_label=torch.nn.functional.one_hot(class_label,num_classes=self.nbr_classes)
        sample['class_label']=class_label
        print(class_label.shape)
        print(sample['fg_channel'].shape)
        return sample


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    args=parser.parse_args()
    trans=transform.Compose([transform.HorizontalFilp(),
                             transform.Crop(base_size=513,crop_height=513,crop_width=513,type='central'),
                             transform.GaussianBlur(),
                             transform.ToTensor(),
                             transform.Normalize(std=[0.229, 0.224, 0.225],
                                               mean=[0.485, 0.456, 0.406])
                             ])

    split='train_aug'
    data_root="D:\\datasets\\VOC\\VOCdevkit\\VOC2012"
    data_list="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train_aug.txt"

    Voc_set=VOC_Dataset(split=split,data_root=data_root,transform=trans)
    col_map=Voc_set.palette
    dataloader = DataLoader(Voc_set, batch_size=10, shuffle=True, num_workers=0)

    val_list=[]
    for ii, sample in enumerate(dataloader):
        gt = sample['label'][0]
        salience=sample['salience'][0]

        if (len(torch.unique(gt))>3):
            if len(val_list)<20:
                gt=colorization.colorization(gt.numpy(),col_map)
                img=sample['img'][0]
                #print(img.max())
                img=transform.DeNormalize(std=[0.229, 0.224, 0.225],
                                               mean=[0.485, 0.456, 0.406])(img)
                gt=transforms.ToTensor()(gt.convert('RGB'))
                salience=transforms.ToTensor()(transforms.ToPILImage()(salience).convert('RGB'))
                val_list.extend([img,gt,salience])
            else:
                break
    val_list=torch.stack(val_list,0)
    val_list=make_grid(val_list,nrow=3,padding=5)
    val_list=transforms.ToPILImage()(val_list)
    val_list.show()
