from torch.utils.data import Dataset,DataLoader
import os
import torch
import numpy as np
from PIL import Image
from random import sample
import json
from torchvision import transforms
from torchvision.utils import make_grid
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
# voc_name_list=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat", "chair",
#                "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# coco_name_list=open(r"../data/osr_splits/coco_label_names.txt").readlines()
#
# voc_known_classes=[0,2,6,7,11,14]
# voc_unknown_classes=[1,3,4,5,8,9,12,13,15,16,17,18,19]
#
# voc_cls_labels_dict = np.load('../data/cls_labels_voc.npy', allow_pickle=True).item()
# coco_cls_labels_dict = np.load('../data/cls_labels_coco.npy', allow_pickle=True).item()

def create_COCO_osr_class_split(name="coco",root_path="D:\\datasets\\coco\\2014\\"):
    print("Creating {0} OSR split".format(name))
    name_list = {}
    cls_label_dict=coco_cls_labels_dict
    train_img_name_list = np.loadtxt(os.path.join(root_path, "train14.txt"), dtype=str)
    val_img_name_list   = np.loadtxt(os.path.join(root_path, "val14.txt"), dtype=str)
    train_img_name_list=["COCO_train2014_"+x for x in train_img_name_list]
    val_img_name_list=["COCO_val2014_"+x for x in val_img_name_list]
    nbr_cls = np.zeros(80)
    for i in range(80):
        name_list[str(i)]=[]
    nbr_known_cls = 16
    nbr_img_per_cls=400
    img_name_list=np.concatenate([train_img_name_list,val_img_name_list],axis=0)
    assert img_name_list.size==len(cls_label_dict)
    for img in img_name_list:
        pre_fix=img.split('_')[1]
        key=int(img.split('_')[-1])
        label=cls_label_dict[key]
        if label.sum()==1:
            nbr_cls[label==1]+=1
            name_list[str((label==1).nonzero()[0][0])].append(pre_fix+"/"+img+".jpg")
    full_cls_set=list((nbr_cls>=50).nonzero()[0])
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])

    print("Would sample {0} training classes and {1} open classes".format(nbr_known_cls,len(full_cls_set)-nbr_known_cls))
    for split in range(5):

        from random import sample
        knwon_cls=list(sample(full_train_cls,nbr_known_cls))
        unknown_cls=list(set(full_cls_set).difference(set(knwon_cls)))
        cls_info={}
        cls_info["known_cls"]=knwon_cls
        cls_info["unknown_cls"]=unknown_cls
        print("Split {0}: train classes {1}, open classes {2} ".format(split+1,knwon_cls,unknown_cls))
        train_file=open("../data/osr_splits/"+name+"/"+str(name)+"_train_split_"+str(split+1)+".txt","w+")
        val_file=open("../data/osr_splits/"+name+"/"+str(name)+"_val_split_"+str(split+1)+".txt","w+")
        test_file = open("../data/osr_splits/"+name+"/" + str(name) + "_test_split_" + str(split + 1) + ".txt", "w+")
        for cls_index,train_cls in enumerate(knwon_cls):
            img_set=name_list[str(train_cls)][:nbr_img_per_cls]
            img_set=[str(x)+" "+str(cls_index)+"\n" for x in img_set]
            train_file.writelines(img_set)
            img_set=name_list[str(train_cls)][-50:]
            img_set=[str(x)+" "+str(cls_index)+"\n" for x in img_set]
            val_file.writelines(img_set)
        for test_cls in unknown_cls:
            img_set=name_list[str(test_cls)][-50:]
            img_set=[str(x)+" -1\n" for x in img_set]
            test_file.writelines(img_set)
        train_file.close()
        val_file.close()
        test_file.close()
        with open("../data/osr_splits/"+name+"/" + str(name) + "_cls_info_split_" + str(split + 1) + ".json", "w") as outfile:
            json.dump(cls_info, outfile,cls=NpEncoder)

def create_VOC_osr_class_split(name="voc",root_path="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\"):
    print("Creating {0} OSR split".format(name))
    name_list = {}
    cls_label_dict=voc_cls_labels_dict
    img_name_list = [np.loadtxt(os.path.join(root_path, "train_aug.txt"), dtype=np.int32),
                     np.loadtxt(os.path.join(root_path, "val.txt"), dtype=np.int32)]
    nbr_cls=np.zeros(20)
    for i in range(20):
        name_list[str(i)]=[]
    nbr_known_cls=5
    nbr_img_per_cls=400
    img_name_list=np.concatenate(img_name_list,axis=0)
    assert img_name_list.size==len(cls_label_dict)
    for img in img_name_list:
        label=cls_label_dict[int(img)]
        if label.sum()==1:
            nbr_cls[label==1]+=1
            name_list[str((label==1).nonzero()[0][0])].append(img)
    full_cls_set=list((nbr_cls>=50).nonzero()[0])
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])
    print("Would sample {0} training classes and {1} open classes".format(nbr_known_cls,len(full_cls_set)-nbr_known_cls))
    for split in range(5):
        cls_info={}

        knwon_cls=list(sample(full_train_cls,nbr_known_cls))
        unknown_cls=list(set(full_cls_set).difference(set(knwon_cls)))
        cls_info["known_cls"]=knwon_cls
        cls_info["unknown_cls"]=unknown_cls
        print("Split {0}: train classes {1}, open classes {2} ".format(split+1,knwon_cls,unknown_cls))
        train_file=open("../data/osr_splits/"+name+"/"+str(name)+"_train_split_"+str(split+1)+".txt","w+")
        val_file=open("../data/osr_splits/"+name+"/"+str(name)+"_val_split_"+str(split+1)+".txt","w+")
        test_file = open("../data/osr_splits/"+name+"/" + str(name) + "_test_split_" + str(split + 1) + ".txt", "w+")
        for cls_index,train_cls in enumerate(knwon_cls):
            img_set=name_list[str(train_cls)][:nbr_img_per_cls]
            img_set=["JPEGImages/"+str(x)+".jpg  "+str(cls_index)+"\n" for x in img_set]
            train_file.writelines(img_set)
            img_set=name_list[str(train_cls)][-50:]
            img_set=["JPEGImages/"+str(x)+".jpg  "+str(cls_index)+"\n" for x in img_set]
            val_file.writelines(img_set)
        for test_cls in unknown_cls:
            img_set=name_list[str(test_cls)][-50:]
            img_set=["JPEGImages/"+str(x)+".jpg  -1\n" for x in img_set]
            test_file.writelines(img_set)
        train_file.close()
        val_file.close()
        test_file.close()
        with open("../data/osr_splits/"+name+"/" + str(name) + "_cls_info_split_" + str(split + 1) + ".json", "w") as outfile:
            json.dump(cls_info, outfile,cls=NpEncoder)



def VOC_osr_class_split(name="voc",root_path="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\"):
    print("Creating {0} OSR split".format(name))
    name_list = {}
    cls_label_dict=voc_cls_labels_dict
    img_name_list = [np.loadtxt(os.path.join(root_path, "train_aug.txt"), dtype=np.int32),
                     np.loadtxt(os.path.join(root_path, "val.txt"), dtype=np.int32)]
    nbr_cls=np.zeros(20)
    for i in range(20):
        name_list[str(i)]=[]
    nbr_known_cls=6
    nbr_img_per_cls=400
    img_name_list=np.concatenate(img_name_list,axis=0)
    assert img_name_list.size==len(cls_label_dict)
    for img in img_name_list:
        label=cls_label_dict[int(img)]
        if label.sum()==1:
            nbr_cls[label==1]+=1
            name_list[str((label==1).nonzero()[0][0])].append(img)
    full_cls_set=list((nbr_cls>=50).nonzero()[0])
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])
    print("Would sample {0} training classes and {1} open classes".format(nbr_known_cls,len(full_cls_set)-nbr_known_cls))
    for split in range(1):
        cls_info={}

        knwon_cls=list(sample(full_train_cls,nbr_known_cls))
        unknown_cls=list(set(full_cls_set).difference(set(knwon_cls)))
        cls_info["known_cls"]=knwon_cls
        cls_info["unknown_cls"]=unknown_cls
        print("Split {0}: train classes {1}, open classes {2} ".format(split+1,knwon_cls,unknown_cls))
        train_file=open("../"+str(name)+"_train.txt","w+")
        val_file=open("../"+str(name)+"_val.txt","w+")
        test_file = open("../"+str(name) + "_test.txt", "w+")
        for cls_index,train_cls in enumerate(knwon_cls):
            img_set=name_list[str(train_cls)][:nbr_img_per_cls]
            img_set=["JPEGImages/"+str(x)+".jpg  "+str(cls_index)+"\n" for x in img_set]
            train_file.writelines(img_set)
            img_set=name_list[str(train_cls)][-50:]
            img_set=["JPEGImages/"+str(x)+".jpg  "+str(cls_index)+"\n" for x in img_set]
            val_file.writelines(img_set)
        for test_cls in unknown_cls:
            img_set=name_list[str(test_cls)][-50:]
            img_set=["JPEGImages/"+str(x)+".jpg  -1\n" for x in img_set]
            test_file.writelines(img_set)
        train_file.close()
        val_file.close()
        test_file.close()
        with open("../" + str(name) + "_cls_info_split.json", "w") as outfile:
            json.dump(cls_info, outfile,cls=NpEncoder)

def COCO_osr_class_split(name="coco",root_path="D:\\datasets\\coco\\2014\\"):
    print("Creating {0} OSR split".format(name))
    name_list = {}
    cls_label_dict=coco_cls_labels_dict
    train_img_name_list = np.loadtxt(os.path.join(root_path, "train14.txt"), dtype=str)
    val_img_name_list   = np.loadtxt(os.path.join(root_path, "val14.txt"), dtype=str)
    train_img_name_list=["COCO_train2014_"+x for x in train_img_name_list]
    val_img_name_list=["COCO_val2014_"+x for x in val_img_name_list]
    nbr_cls = np.zeros(80)
    for i in range(80):
        name_list[str(i)]=[]
    nbr_known_cls = 20
    nbr_img_per_cls=400
    img_name_list=np.concatenate([train_img_name_list,val_img_name_list],axis=0)
    assert img_name_list.size==len(cls_label_dict)
    for img in img_name_list:
        pre_fix=img.split('_')[1]
        key=int(img.split('_')[-1])
        label=cls_label_dict[key]
        if label.sum()==1:
            nbr_cls[label==1]+=1
            name_list[str((label==1).nonzero()[0][0])].append(pre_fix+"/"+img+".jpg")
    full_cls_set=list((nbr_cls>=50).nonzero()[0])
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])

    print("Would sample {0} training classes and {1} open classes".format(nbr_known_cls,len(full_cls_set)-nbr_known_cls))
    for split in range(1):

        from random import sample
        knwon_cls=list(sample(full_train_cls,nbr_known_cls))
        unknown_cls=list(set(full_cls_set).difference(set(knwon_cls)))
        cls_info={}
        cls_info["known_cls"]=knwon_cls
        cls_info["unknown_cls"]=unknown_cls
        print("Split {0}: train classes {1}, open classes {2} ".format(split+1,knwon_cls,unknown_cls))
        train_file=open("../"+str(name)+"_train_split.txt","w+")
        val_file=open("../"+str(name)+"_val_split.txt","w+")
        test_file = open("../" + str(name) + "_test_split.txt", "w+")
        for cls_index,train_cls in enumerate(knwon_cls):
            img_set=name_list[str(train_cls)][:nbr_img_per_cls]
            img_set=[str(x)+" "+str(cls_index)+"\n" for x in img_set]
            train_file.writelines(img_set)
            img_set=name_list[str(train_cls)][-50:]
            img_set=[str(x)+" "+str(cls_index)+"\n" for x in img_set]
            val_file.writelines(img_set)
        for test_cls in unknown_cls:
            img_set=name_list[str(test_cls)][-50:]
            img_set=[str(x)+" -1\n" for x in img_set]
            test_file.writelines(img_set)
        train_file.close()
        val_file.close()
        test_file.close()
        with open("../" + str(name) + "_cls_info_split.json", "w") as outfile:
            json.dump(cls_info, outfile,cls=NpEncoder)
class OSR_dataset(Dataset):
    def __init__(self, data_root="D:\\datasets\\VOC\\VOCdevkit\\",prefix='train',exp="voc", transform=None):
        self.root_dir=data_root
        self.transform=transform
        self.prefix=prefix
        self.exp=exp
        self.name_list=self.__get_name_list()
        self.transform=transform
        self.max_num_object=1

        if exp=="voc":
            self.num_known_classes=6
        elif exp=="coco":
            self.num_known_classes=20

    def __get_name_list(self):
        text_file=os.path.join("./data/osr_splits/"+self.exp+"/"+self.exp+"_"+self.prefix+".txt")
        line_list=open(text_file).readlines()
        print("Totally have {} samples in {} set.".format(len(line_list),self.prefix))
        img_path=[]
        labels=[]
        if self.exp=='voc':
            aug_str='_'
        else:
            aug_str=''
        for idx,line in enumerate(line_list):
            name=line.split()[0]
            img_path.append(name[:15]+aug_str+name[15:])
            labels.append(line.split()[1])
        self.labels=labels
        return img_path

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name=self.name_list[index]
        sample={}
        img_path=os.path.join(self.root_dir,name)
        image=Image.open(img_path).convert("RGB")
        #print(np.asarray(image).shape)
        image=self.transform(image)
        label=int(self.labels[index])
        label=torch.as_tensor(label,dtype=torch.long)
        sample['img']=image
        sample['label']=label

        if self.prefix in ["train","val"]:
            class_label=torch.zeros((self.max_num_object))
            class_label[0]=label
            class_label=torch.as_tensor(class_label,dtype=torch.int64)

            class_label=torch.nn.functional.one_hot(class_label,num_classes=self.num_known_classes)
            sample['class_label']=class_label

        return sample


def Get_OSR_Datasets(train_transform, test_transform,dataroot="D:\\datasets\\VOC\\",exp="voc"):

    train_dataset = OSR_dataset(prefix='train',data_root=dataroot,exp=exp,transform=train_transform)
    val_dataset = OSR_dataset(prefix='val',data_root=dataroot,exp=exp,transform=test_transform)
    test_dataset = OSR_dataset(prefix='test',data_root=dataroot,exp=exp,transform=test_transform)
    #mix_known_unknown_dataset = OSR_dataset(split='test_mixture',data_root=dataroot,exp=exp,transform=test_transform)

    print('Train: ', len(train_dataset), 'Val: ', len(val_dataset), 'Single_Out: ', len(test_dataset))

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    return all_datasets

if __name__=="__main__":
    #create_COCO_osr_class_split(name="COCO",root_path="D:\\datasets\\coco\\2014\\")
    #create_VOC_osr_class_split(name="voc",root_path="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\")
    #VOC_osr_class_split(name="voc",root_path="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\")
    COCO_osr_class_split(name="COCO",root_path="D:\\datasets\\coco\\2014\\")
    # from utils import transform
    # from torchvision import transforms
    # trans=transforms.Compose([
    #                          transforms.Resize((224,224)),
    #                          transforms.ToTensor(),
    #                          transforms.Normalize(std=[0.229, 0.224, 0.225],
    #                                            mean=[0.485, 0.456, 0.406])
    #                          ])
    # #datasets=Get_OSR_Datasets(train_transform=trans,test_transform=trans,dataroot="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\",exp="voc")
    # datasets=Get_OSR_Datasets(train_transform=trans,test_transform=trans,dataroot="D:\\datasets\\coco\\2014\\",exp="coco")
    # dataloaders = {}
    # for k, v, in datasets.items():
    #     shuffle = True if k == 'train' else False
    #     dataloaders[k] = DataLoader(v, batch_size=8,
    #                                 shuffle=shuffle, sampler=None, num_workers=0)
    # #print(datasets['train'][0])
    # trainloader = dataloaders['train']
    # testloader = dataloaders['val']
    # outloader = dataloaders['test']
    #
    # for loader in [outloader]:
    #     val_list = []
    #     for ii, sample in enumerate(loader):
    #         if len(val_list)<=15:
    #             img,label=sample['img'][0].cuda(),sample['label'][0].cuda()
    #             img = transform.DeNormalize(std=[0.229, 0.224, 0.225],
    #                                         mean=[0.485, 0.456, 0.406])(img)
    #             print(coco_name_list[label])
    #             val_list.append(img)
    #         else:
    #             break
    # val_list = torch.stack(val_list, 0)
    # val_list = make_grid(val_list, nrow=1, padding=5)
    # val_list = transforms.ToPILImage()(val_list)
    # val_list.show()
