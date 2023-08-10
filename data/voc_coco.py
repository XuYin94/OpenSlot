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

voc_name_list=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

train_name=["aeroplane","bird","bus","cat","dog","person","train"]
train_classes=[0,2,6,7,11,14,18]
unknown_classes=[1,3,4,5,8,9,10,12,13,15,16,17,19]
#class_overlap_coco_indices=[5,2,15,9,40,6,3,16,57,20,61,17,18,4,1,59,19,58,7,63]

def test():
    voc_class_label=np.load("../data/cls_labels.npy",allow_pickle=True).item()
    #coco_class_label=np.load("../data/cls_labels_coco.npy",allow_pickle=True).item()
    #print(coco_class_label)
    #voc_dict={}
    #coco_dict={}
    voc_name_list=open("D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\list\\train_aug.txt").readlines()+open("D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\list\\val.txt").readlines()
    #coco_name_list=open("D:\\datasets\\coco\\2014\\coco_multi_train.txt").readlines()+open("D:\\datasets\\coco\\2014\\coco_multi_val.txt").readlines()
    #voc_nbr=np.zeros(20)
    #coco_nbr=np.zeros(20)
    name_dict=[[],[],[],[],[],[],[]]
    mix_known_unknown=[]
    single_unknown=[]
    #name_dict.append([] for i in range(7))
    for line in voc_name_list:
        line = line.strip()
        line_split = line.split(' ')
        img_name = line_split[0].split('/')[1][:-4]
        img_label=voc_class_label[int(img_name)]
        if np.sum(img_label)==1.0:
            class_indices=(img_label==1.0).nonzero()
            #print(class_indices)
            if class_indices[0] in train_classes:
                indices=(train_classes==class_indices[0]).nonzero()[0]
                name_dict[indices[0]].append(img_name)
            else:
                single_unknown.append(img_name)
        else:
            mix_known_unknown.append(img_name)
    train_list=np.concatenate([x[:400] for x in name_dict],axis=0)
    val_list=np.concatenate([x[401:] for x in name_dict],axis=0)
    single_unknown=np.asarray(single_unknown)
    mix_known_unknown=np.asarray(mix_known_unknown)
    np.savetxt('../data/train_list.txt',train_list,fmt='%s')
    np.savetxt('../data/val_list.txt',val_list,fmt='%s')
    np.savetxt('../data/single_unknown_list.txt',single_unknown,fmt='%s')
    np.savetxt('../data/mix_known_unknown_list.txt',mix_known_unknown,fmt='%s')
    # for line in coco_name_list:
    #     img_name = line.strip()
    #     img_label = coco_class_label[int(img_name)]
    #     #print(img_label.shape)
    #     if np.sum(img_label) == 1.0:
    #         class_indices = (img_label == 1.0).nonzero()[0]+1
    #         if class_indices in class_overlap_coco_indices:
    #             overlapped_indices=np.where(class_overlap_coco_indices==class_indices)
    #             coco_nbr[overlapped_indices]+=1
    #print(voc_nbr)
class OSR_dataset(Dataset):
    def __init__(self, split='train', data_root="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\",  transform=None):
        self.root_dir=data_root
        self.transform=transform
        self.split=split
        self.name_list=self.__get_name_list(self.split)
        self.class_label=np.load("../data/cls_labels.npy",allow_pickle=True).item()
        self.transform=transform

    def __get_name_list(self,split):
        assert split in ['train','val','single_unknown','mix_known_unknown']
        #name_list=[]
        text_file=os.path.join("../data/"+split+"_list.txt")
        line_list=open(text_file).readlines()
        print("Totally have {} samples in {} set.".format(len(line_list),self.split))
        return line_list


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name=self.name_list[index].strip()
        image=Image.open(self.root_dir+"/JPEGImages/"+name+".jpg")
        image=self.transform({'img':image,'label':image,'salience':image})['img']
        if self.split in ['train','val']:
            gt=torch.zeros(7)
            label=self.class_label[int(self.name_list[index])].nonzero()[0]
            indices=(train_classes==label).nonzero()
            gt[indices]=1
            return {'img':image,'name':self.name_list[index],'label':gt}
        else:
            return image

def get_osr_voc_datasets():
    trans=transform.Compose([transform.HorizontalFilp(),
                             transform.Crop(base_size=513,crop_height=513,crop_width=513,type='central'),
                             transform.GaussianBlur(),
                             transform.ToTensor(),
                             transform.Normalize(std=[0.229, 0.224, 0.225],
                                               mean=[0.485, 0.456, 0.406])
                             ])
    train_dataset=OSR_dataset(split='train',transform=trans)
    val_dataset=OSR_dataset(split='val')
    single_unknown_dataset=OSR_dataset(split='single_unknown')
    mix_known_unknown_dataset=OSR_dataset(split='mix_known_unknown')
    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'single_unknown': single_unknown_dataset,
        'mix_unknown': mix_known_unknown_dataset,
    }

    return all_datasets


if __name__=="__main__":
    #test()
    x = get_osr_voc_datasets()
    #print([len(v) for k, v in x.items()])
    dataset=x['train']
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

    val_list=[]
    for ii, sample in enumerate(dataloader):
        if len(val_list) < 20:
            gt = sample['label'][0]
            name=train_name[(gt==1).nonzero()]
            img = sample['img'][0]
            img = transform.DeNormalize(std=[0.229, 0.224, 0.225],
                                        mean=[0.485, 0.456, 0.406])(img)
            print(name)
            val_list.extend([img])
        else:
            break
val_list = torch.stack(val_list, 0)
val_list = make_grid(val_list, nrow=1, padding=5)
val_list = transforms.ToPILImage()(val_list)
val_list.show()
