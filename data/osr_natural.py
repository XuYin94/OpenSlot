from torch.utils.data import Dataset,DataLoader
import os
import torch
import numpy as np
from PIL import Image
from random import sample
import torchvision
import json
from torchvision import transforms
from torchvision.utils import make_grid
def Get_cifar_Dataset(dataroot="/root/yinxu/Dataset/",split_idx="1"):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(std=[0.229, 0.224, 0.225],
                             mean=[0.485, 0.456, 0.406])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        # transforms.CenterCrop((224, 224)),
        transforms.Normalize(std=[0.229, 0.224, 0.225],
                             mean=[0.485, 0.456, 0.406])
    ])

    train_dataset = Tiny_dataset(prefix='train', data_root=dataroot, split_idx=split_idx,nbr_classes=50,transform=train_transform)
    known_test_dataset = Tiny_dataset(prefix='known_test', data_root=dataroot, split_idx=split_idx,nbr_classes=50,transform=test_transform)
    unknown_test_dataset = Tiny_dataset(prefix='unknown_test', data_root=dataroot, split_idx=split_idx,nbr_classes=50,transform=test_transform)


    print('Train: ', len(train_dataset), 'Test: ', len(known_test_dataset),
          "Unknow_test:",len(unknown_test_dataset))
    all_datasets = {
        'train': train_dataset,
        'test_known': known_test_dataset,
        'test_unknown': unknown_test_dataset
    }

    return all_datasets


def Get_Tiny_Dataset(dataroot="/root/yinxu/Dataset/",split_idx="1"):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(std=[0.229, 0.224, 0.225],
                             mean=[0.485, 0.456, 0.406])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        # transforms.CenterCrop((224, 224)),
        transforms.Normalize(std=[0.229, 0.224, 0.225],
                             mean=[0.485, 0.456, 0.406])
    ])

    train_dataset = Cifar_dataset(prefix='train', data_root=dataroot, split_idx=split_idx,nbr_classes=50,transform=train_transform)
    known_test_dataset =Cifar_dataset(prefix='known_test', data_root=dataroot, split_idx=split_idx,nbr_classes=50,transform=test_transform)
    unknown_test_dataset = Cifar_dataset(prefix='unknown_test', data_root=dataroot, split_idx=split_idx,nbr_classes=50,transform=test_transform)


    print('Train: ', len(train_dataset), 'Test: ', len(known_test_dataset),
          "Unknow_test:",len(unknown_test_dataset))
    all_datasets = {
        'train': train_dataset,
        'test_known': known_test_dataset,
        'test_unknown': unknown_test_dataset
    }

    return all_datasets

class Tiny_dataset(Dataset):
    def __init__(self, data_root="D:\\datasets\\openood\\data\\image_classic\\",prefix="train",nbr_classes=20,split_idx=1,transform=None):
        self.root_dir=data_root
        self.transform=transform

        self.transform=transform
        self.prefix=prefix
        self.max_num_object=1
        #if self.prefix=="train":
        self.num_known_classes=nbr_classes
        self.split_idx=split_idx
        self.name_list=self.__get_name_list()
    def __get_name_list(self):
        if self.prefix=="train":
            text_file=os.path.join("./data/osr_splits/osr_cifar6/"+self.prefix+"/"+self.prefix+"_cifar10_"+str(self.num_known_classes)+"_seed"+str(self.split_idx)+".txt")
        elif self.prefix=="known_test":
            text_file = os.path.join("./data/osr_splits/osr_cifar6/test/test_cifar10_" + str(
                self.num_known_classes) + "_id_seed" + str(self.split_idx) + ".txt")
        else:
            text_file = os.path.join("./data/osr_splits/osr_cifar6/test/test_cifar10_" + str(
                self.num_known_classes) + "_ood_seed" + str(self.split_idx) + ".txt")
        print(text_file)
        line_list=open(text_file).readlines()
        print("Totally have {} samples in {} set.".format(len(line_list),self.prefix))
        img_path=[]
        labels=[]
        for idx,line in enumerate(line_list):
            name=line.strip().split()[0]
            img_path.append(name)
            labels.append(line.strip().split()[1])
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

        if self.prefix in ["train"]:
            class_label=torch.zeros((self.max_num_object))
            selected_slots=torch.zeros((self.max_num_object))
            class_label[0]=label
            selected_slots[0]=1
            class_label=torch.as_tensor(class_label,dtype=torch.int64)

            class_label=torch.nn.functional.one_hot(class_label,num_classes=self.num_known_classes)
            sample['class_label']=class_label

            sample['fg_channel']=selected_slots.unsqueeze(1)


        return sample



class Cifar_dataset(Dataset):
    def __init__(self, data_root="D:\\datasets\\openood\\data\\image_classic\\",prefix="train",nbr_classes=20,split_idx=1,transform=None):
        self.root_dir=data_root
        self.transform=transform

        self.transform=transform
        self.prefix=prefix
        self.max_num_object=1
        #if self.prefix=="train":
        self.num_known_classes=nbr_classes
        self.split_idx=split_idx
        self.name_list=self.__get_name_list()
    def __get_name_list(self):
        if self.prefix=="train":
            text_file=os.path.join("./data/osr_splits/osr_cifar50/"+self.prefix+"/"+self.prefix+"_cifar100_"+str(self.num_known_classes)+"_seed"+str(self.split_idx)+".txt")
        elif self.prefix=="known_test":
            text_file = os.path.join("./data/osr_splits/osr_cifar50/test/test_cifar100_" + str(
                self.num_known_classes) + "_id_seed" + str(self.split_idx) + ".txt")
        else:
            text_file = os.path.join("./data/osr_splits/osr_cifar50/test/test_cifar100_" + str(
                self.num_known_classes) + "_ood_seed" + str(self.split_idx) + ".txt")
        print(text_file)
        line_list=open(text_file).readlines()
        print("Totally have {} samples in {} set.".format(len(line_list),self.prefix))
        img_path=[]
        labels=[]
        for idx,line in enumerate(line_list):
            name=line.strip().split()[0]
            img_path.append(name)
            labels.append(line.strip().split()[1])
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

        if self.prefix in ["train"]:
            class_label=torch.zeros((self.max_num_object))
            selected_slots=torch.zeros((self.max_num_object))
            class_label[0]=label
            selected_slots[0]=1
            class_label=torch.as_tensor(class_label,dtype=torch.int64)

            class_label=torch.nn.functional.one_hot(class_label,num_classes=self.num_known_classes)
            sample['class_label']=class_label

            sample['fg_channel']=selected_slots.unsqueeze(1)


        return sample



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def create_COCO_osr_class_split(name="coco",root_path="D:\\datasets\\coco\\2014\\"):
    print("Creating {0} OSR split".format(name))
    name_list = {}
    cls_label_dict=coco_cls_labels_dict
    train_img_name_list = np.loadtxt(os.path.join(root_path, "coco_multi_train.txt"), dtype=str)
    val_img_name_list   = np.loadtxt(os.path.join(root_path, "coco_multi_val.txt"), dtype=str)
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
    img_name_list = [np.loadtxt(os.path.join(root_path, "voc_multi_train.txt"), dtype=np.int32),
                     np.loadtxt(os.path.join(root_path, "voc_multi_val.txt"), dtype=np.int32)]
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
            self.num_slot=6
        elif exp=="coco":
            self.num_known_classes=20
            self.num_slot=7

    def __get_name_list(self):
        text_file=os.path.join("./data/osr_splits/"+self.exp+"/single/"+self.exp+"_"+self.prefix+".txt")
        line_list=open(text_file).readlines()
        print("Totally have {} samples in {} set.".format(len(line_list),self.prefix))
        img_path=[]
        labels=[]
        # if self.exp=='voc':
        #     aug_str='_'
        # else:
        #     aug_str=''
        for idx,line in enumerate(line_list):
            name=line.split()[0]
            img_path.append(name)
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

        if self.prefix in ["train"]:
            class_label=torch.zeros((self.max_num_object))
            selected_slots=torch.zeros((self.max_num_object))
            class_label[0]=label
            selected_slots[0]=1
            class_label=torch.as_tensor(class_label,dtype=torch.int64)

            class_label=torch.nn.functional.one_hot(class_label,num_classes=self.num_known_classes)
            sample['class_label']=class_label
            sample['fg_channel']=selected_slots.unsqueeze(1)

           #bg_class_label=torch.zeros((self.num_slot))
            #bg_class_label[1:]=1
            #sample['bg_channel']=bg_class_label.unsqueeze(1)

            #bg_class_label=torch.nn.functional.one_hot(torch.as_tensor(bg_class_label,dtype=torch.int64),num_classes=2)
            #bg_class_label = torch.as_tensor(bg_class_label, dtype=torch.float64).unsqueeze(1)

            #sample['bg_class_label']=bg_class_label
        return sample


def Get_OSR_Datasets(train_transform, test_transform,dataroot="D:\\datasets\\VOC\\",exp="voc"):

    train_dataset = OSR_dataset(prefix='train',data_root=dataroot,exp=exp,transform=train_transform)
    val_dataset = OSR_dataset(prefix='val',data_root=dataroot,exp=exp,transform=test_transform)


    single_unknown_dataset = OSR_dataset(prefix='no_mixture_test',data_root=dataroot,exp=exp,transform=test_transform)
    all_mixture_unknown_dataset = OSR_dataset(prefix='all_mixture_test',data_root=dataroot,exp=exp,transform=test_transform)

    openness_easy_mixture_unknown_dataset = OSR_dataset(prefix='openness_easy_test',data_root=dataroot,exp=exp,transform=test_transform)
    openness_hard_mixture_unknown_dataset = OSR_dataset(prefix='openness_hard_test',data_root=dataroot,exp=exp,transform=test_transform)

    dominance_easy_mixture_unknown_dataset = OSR_dataset(prefix='dominance_easy_test',data_root=dataroot,exp=exp,transform=test_transform)
    dominance_hard_mixture_unknown_dataset = OSR_dataset(prefix='dominance_hard_test',data_root=dataroot,exp=exp,transform=test_transform)


    print('Train: ', len(train_dataset), 'Test: ', len(val_dataset), 'Single_Out: ',
          len(single_unknown_dataset),'All_mixture: ',
          len(all_mixture_unknown_dataset),'Openness_easy_mixture:', len(openness_easy_mixture_unknown_dataset),'Openness_hard_mixture:', len(openness_hard_mixture_unknown_dataset),'Dominance_easy_mixture:', len(dominance_easy_mixture_unknown_dataset),'Dominance_hard_mixture:', len(dominance_hard_mixture_unknown_dataset))
    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'No_mixture': single_unknown_dataset,
        'all_mixture': all_mixture_unknown_dataset,
        'openness_easy_mixture_test':openness_easy_mixture_unknown_dataset,
        'openness_hard_mixture_test': openness_hard_mixture_unknown_dataset,
        'dominance_easy_mixture_test':dominance_easy_mixture_unknown_dataset,
        'dominance_hard_mixture_test': dominance_hard_mixture_unknown_dataset
    }

    return all_datasets

class voc_detection(Dataset):
    def __init__(self, data_root="D:\\datasets\\VOC\\VOCdevkit\\",prefix='train',exp="voc_detection", transform=None):
        self.root_dir=data_root
        self.transform=transform
        self.prefix=prefix
        self.exp=exp
        self.exp_info='./data/osr_splits/'+str(exp)+'/'+prefix+''
        self.img_list=open(r'' + self.exp_info + '_set.txt').readlines()
        self.cls_labels_dict = np.load(self.exp_info+"_label.npy")
        
        self.num_known_classes=20
        self.transform=transform
        self.max_num_object = 7

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        name=self.img_list[index].strip()
        #print(name)
        sample={}
        img_path=os.path.join(self.root_dir,name)

        image=Image.open(img_path).convert("RGB")
        image=self.transform(image)
        sample['img']=image
        if self.prefix in ["train","val"]:

            sample['label'] = self.cls_labels_dict[index]
            label=self.cls_labels_dict[index].nonzero()[0][:self.max_num_object]
            nbr_clss=self.max_num_object if len(label)>self.max_num_object else len(label)
            selected_slots = torch.zeros((self.max_num_object))
            semantic_label = torch.zeros((self.max_num_object))
            selected_slots[:nbr_clss] = 1
            semantic_label[:nbr_clss] = torch.as_tensor(np.array(label),dtype=torch.long)+1

            sample['fg_channel']=selected_slots.unsqueeze(1)
            semantic_label=torch.as_tensor(semantic_label,dtype=torch.int64)
            semantic_label=torch.nn.functional.one_hot(semantic_label,num_classes=self.num_known_classes+1)[:,1:]
            sample['class_label']=semantic_label
        else:
            sample['label']=torch.tensor(float(np.inf))
        return sample
class intra_coco_multi_label(Dataset):
    def __init__(self, data_root="D:\\datasets\\VOC\\VOCdevkit\\",prefix='train',exp="voc", transform=None):
        self.root_dir=data_root
        self.transform=transform
        self.prefix=prefix
        self.exp=exp
        self.exp_info='./data/osr_splits/multi_osr/Intra_coco/'+str(exp)+'/'+prefix+''
        self.img_list=open(r'' + self.exp_info + '_set.txt').readlines()
        if prefix in ["train"]:
            self.cls_labels_dict = np.load(self.exp_info+"_label.npy")
        
        
        self.transform=transform
        self.max_num_object = 7
        if exp=="Task1":
            self.num_known_classes=20
        elif exp=="Task2":
            self.num_known_classes=40
        else:
            self.num_known_classes=60

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        name=self.img_list[index].strip()
        if self.prefix=="no_mixture":
            name=name[:-2]

        #print(name)
        sample={}
        img_path=os.path.join(self.root_dir,name)

        image=Image.open(img_path).convert("RGB")
        image=self.transform(image)
        sample['img']=image
        if self.prefix in ["train"]:

            sample['label'] = self.cls_labels_dict[index]
            label=self.cls_labels_dict[index].nonzero()[0][:self.max_num_object]
            nbr_clss=self.max_num_object if len(label)>self.max_num_object else len(label)
            selected_slots = torch.zeros((self.max_num_object))
            semantic_label = torch.zeros((self.max_num_object))
            selected_slots[:nbr_clss] = 1
            semantic_label[:nbr_clss] = torch.as_tensor(np.array(label),dtype=torch.long)+1

            sample['fg_channel']=selected_slots.unsqueeze(1)
            semantic_label=torch.as_tensor(semantic_label,dtype=torch.int64)
            semantic_label=torch.nn.functional.one_hot(semantic_label,num_classes=self.num_known_classes+1)[:,1:]
            sample['class_label']=semantic_label
        else:
            sample['label']=torch.tensor(float(np.inf))
        return sample


class Multi_label_OSR_dataset(Dataset):
    def __init__(self, data_root="D:\\datasets\\VOC\\VOCdevkit\\",prefix='train',exp="voc", transform=None):
        self.root_dir=data_root
        self.transform=transform
        self.prefix=prefix
        self.exp=exp
        if exp=="multi_osr":
            self.exp_info='./data/osr_splits/'+exp+'/voc2coco2014/voc2coco14_osr_'+prefix+''
            print(self.exp_info)
        else:
            self.exp_info='./data/osr_splits/'+exp+'/multi/'+exp+'_multi_'+prefix+''

        self.img_list=open(r'' + self.exp_info + '.txt').readlines()
        if prefix in ["train","train_aug","val"]:
            self.cls_labels_dict = np.load(self.exp_info+"_label.npy")

        self.transform=transform

        if exp=="voc":
            self.max_num_object = 6
            self.num_known_classes=20
        else:
            self.max_num_object = 7
            self.num_known_classes=80

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        name=self.img_list[index].strip()
        #print(name)
        sample={}
        img_path=os.path.join(self.root_dir,name)

        image=Image.open(img_path).convert("RGB")
        image=self.transform(image)
        sample['img']=image
        #print(img_path)
        if self.prefix in ["train","train_aug","val"]:

            sample['label'] = self.cls_labels_dict[index]
            label=self.cls_labels_dict[index].nonzero()[0][:self.max_num_object]
            nbr_clss=self.max_num_object if len(label)>self.max_num_object else len(label)
            selected_slots = torch.zeros((self.max_num_object))
            semantic_label = torch.zeros((self.max_num_object))
            selected_slots[:nbr_clss] = 1
            semantic_label[:nbr_clss] = torch.as_tensor(np.array(label),dtype=torch.long)+1

            sample['fg_channel']=selected_slots.unsqueeze(1)
            semantic_label=torch.as_tensor(semantic_label,dtype=torch.int64)
            semantic_label=torch.nn.functional.one_hot(semantic_label,num_classes=self.num_known_classes+1)[:,1:]
            sample['class_label']=semantic_label
            # print(semantic_label)
            # print(sample["label"])
            # print(sample["fg_channel"])
            #bg_class_label=torch.zeros((self.max_num_object))
            #if nbr_clss<self.max_num_object:
                #nbr_bg_slot=self.max_num_object-nbr_clss
                #bg_class_label[:nbr_bg_slot]=1
            #else:
            #bg_class_label[:3]=1
            #sample['bg_channel']=bg_class_label.unsqueeze(1)
            #bg_class_label = torch.as_tensor(bg_class_label, dtype=torch.float64).unsqueeze(1)

            #sample['bg_class_label']=bg_class_label
        else:
            sample['label']=torch.tensor(float(np.inf))
        return sample


voc2coco_map=[4,1,15,8,43,5,2,16,61,20,66,17,18,3,0,63,19,62,6,71]

def get_non_overlap_coco_img():
    coco_label_dict=np.concatenate((np.load("./osr_splits/coco/multi/coco_multi_train_label.npy"),
                                       np.load("./osr_splits/coco/multi/coco_multi_val_label.npy")),0)
    non_overplapped_cls=list(set(range(80)).difference(set(voc2coco_map)))
    img_list=open(r'./osr_splits/coco/multi/coco17_multi_train.txt').readlines()+open(r'./osr_splits/coco/multi/coco17_multi_val.txt').readlines()


    no_mixture_file=open("../voc2coco17_osr_no_mixture.txt", "w+")
    easy_file = open("../voc2coco17_osr_easy.txt", "w+")
    #midium_file = open("../voc2coco17_osr_midium.txt", "w+")
    hard_file = open("../voc2coco17_osr_hard.txt", "w+")

    for i in range(coco_label_dict.shape[0]):
        voc_cls_sum=coco_label_dict[i,voc2coco_map].sum()
        non_overlap_cls_sum=coco_label_dict[i,non_overplapped_cls].sum()
        name=img_list[i].strip()
        if non_overlap_cls_sum>0: ## contain at least one unknown class
            wildness=voc_cls_sum/non_overlap_cls_sum
            if wildness==0:
                no_mixture_file.writelines(name + "\n")
            elif wildness<1:
                easy_file.writelines(name + "\n")
            elif wildness>1:
                print(wildness)
                hard_file.writelines(name+"\n")
            else:
                continue
        else:
            continue




if __name__=="__main__":
    #get_non_overlap_coco_img()
    # voc_name_list = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    #                  "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    #                  "train", "tvmonitor"]
    # coco_name_list = open(r"../data/osr_splits/coco_label_names.txt").readlines()
    #
    # voc_known_classes = [0, 2, 6, 7, 11, 14]
    # voc_unknown_classes = [1, 3, 4, 5, 8, 9, 12, 13, 15, 16, 17, 18, 19]
    #
    # voc_cls_labels_dict = np.load('../data/cls_labels_voc.npy', allow_pickle=True).item()
    # coco_cls_labels_dict = np.load('../data/cls_labels_coco.npy', allow_pickle=True).item()
    # #create_COCO_osr_class_split(name="COCO",root_path="D:\\datasets\\coco\\2014\\")
    # #create_VOC_osr_class_split(name="voc",root_path="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\")
    # #VOC_osr_class_split(name="voc",root_path="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\")
    # #COCO_osr_class_split(name="COCO",root_path="D:\\datasets\\coco\\2014\\")
    trans=transforms.Compose([
                             transforms.Resize((224,224)),
                             transforms.ToTensor(),
                             transforms.Normalize(std=[0.229, 0.224, 0.225],
                                               mean=[0.485, 0.456, 0.406])
                             ])
    test_dataset=Multi_label_OSR_dataset(data_root="D:\\datasets\\coco\\2017\\",prefix='no_mixture',exp="multi_osr", transform=trans)
    #dataset=Multi_label_OSR_dataset(data_root="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\",prefix='train',exp="voc", transform=trans)
    dataloader=DataLoader(test_dataset, batch_size=8,shuffle=True, sampler=None, num_workers=0)
    #
    from utils import transform
    # #from torchvision i"multi_osr"mport transforms
    #
    # # #datasets=Get_OSR_Datasets(train_transform=trans,test_transform=trans,dataroot="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\",exp="voc")
    # # datasets=Get_OSR_Datasets(train_transform=trans,test_transform=trans,dataroot="D:\\datasets\\coco\\2014\\",exp="coco")
    # # dataloaders = {}
    # # for k, v, in datasets.items():
    # #     shuffle = True if k == 'train' else False
    # #     dataloaders[k] = DataLoader(v, batch_size=8,
    # #                                 shuffle=shuffle, sampler=None, num_workers=0)
    # # #print(datasets['train'][0])
    # # trainloader = dataloaders['train']
    # # testloader = dataloaders['val']
    # # outloader = dataloaders['test']
    # #
    for loader in [dataloader]:
        val_list = []
        for ii, sample in enumerate(loader):
            if len(val_list)<=15:
                img=sample['img'][0].cuda()
                label=sample['label'][0].cuda()
                img=transform.DeNormalize(std=[0.229, 0.224, 0.225],
                                               mean=[0.485, 0.456, 0.406])(img)
                val_list.append(img)
            else:
                break
    val_list = torch.stack(val_list, 0)
    val_list = make_grid(val_list, nrow=1, padding=5)
    val_list = transforms.ToPILImage()(val_list)
    val_list.show()
