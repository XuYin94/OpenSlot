from torch.utils.data import Dataset,DataLoader
import os
import torch
import numpy as np
from PIL import Image
import random
import json
from torchvision import transforms
from torchvision.utils import make_grid

voc_name_list=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
coco_name_list=open(r"../data/osr_splits/coco_label_names.txt").readlines()

voc_known_classes=[0,2,6,7,11,14]
voc_unknown_classes=[1,3,4,5,8,9,12,13,15,16,17,18,19]

voc_cls_labels_dict = np.load('../data/cls_labels_voc.npy', allow_pickle=True).item()
coco_cls_labels_dict = np.load('../data/cls_labels_coco.npy', allow_pickle=True).item()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def multiple_non_repeated_samples(input_list, sample_size, num_samples):
    if len(input_list) <= sample_size:
        return [input_list for _ in range(num_samples)]

    sampled_results = []

    for _ in range(num_samples):
        if len(input_list) < sample_size:
            break

        # Shuffle the input list before each sampling
        random.shuffle(input_list)

        # Take the first 'sample_size' elements as the sample
        sampled_elements = input_list[:sample_size]
        sampled_results.append(sampled_elements)

    return sampled_results
def create_VOC_osr_class_split(name="voc",root_path="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\"):
    print("Creating {0} OSR split".format(name))
    name_list = {}
    mixed_name_list=[]
    cls_label_dict=voc_cls_labels_dict
    img_name_list = [np.loadtxt(os.path.join(root_path, "voc_multi_train.txt"), dtype=float).astype(int),
                     np.loadtxt(os.path.join(root_path, "voc_multi_val.txt"), dtype=float).astype(int)]
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
        else:
            mixed_name_list.append(img)
    full_cls_set=list((nbr_cls>=50).nonzero()[0]) ## number of the object categories that size over 50
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])## number of the object categories that size over 450
    print(full_train_cls)
    print("Would sample {0} training classes and {1} open classes".format(nbr_known_cls,len(full_cls_set)-nbr_known_cls))
    sampled_known_cls=random.sample(full_train_cls,5)
    for split in range(5):
        cls_info={}

        index=full_train_cls.index(sampled_known_cls[split])
        knwon_cls=full_train_cls[:index]+full_train_cls[index+1:]
        knwon_cls.sort()
        unknown_cls=list(set(full_cls_set).difference(set(knwon_cls)))
        unknown_cls.sort()
        cls_info["known_cls"]=knwon_cls
        cls_info["unknown_cls"]=unknown_cls
        print("Split {0}: train classes {1}, open classes {2} ".format(split+1,knwon_cls,unknown_cls))
        train_file=open("../data/osr_splits/"+name+"/split/"+str(name)+"_train_split_"+str(split+1)+".txt","w+")
        val_file=open("../data/osr_splits/"+name+"/split/"+str(name)+"_val_split_"+str(split+1)+".txt","w+")
        test_single_file = open("../data/osr_splits/"+name+"/split/" + str(name) + "_single_test_split_" + str(split + 1) + ".txt", "w+")
        test_easy_mixture_file = open("../data/osr_splits/"+name+"/split/" + str(name) + "_easy_mixture_test_split_" + str(split + 1) + ".txt", "w+")
        test_hard_mixture_file = open("../data/osr_splits/"+name+"/split/" + str(name) + "_hard_mixture_test_split_" + str(split + 1) + ".txt", "w+")
        easy_list=[]
        hard_list=[]
        for img in mixed_name_list:
            label = cls_label_dict[int(img)]
            if label[unknown_cls].sum() == 1 and label[knwon_cls].sum() == 1: ## easy mixture
                easy_list.append(img)
            elif (label[unknown_cls].sum()>=1) and label[unknown_cls].sum()<label[knwon_cls].sum(): ## hard mixture
                hard_list.append(img)
        if len(hard_list)<len(unknown_cls)*50:
            size_mixture_set=len(hard_list)
        img_set=["JPEGImages/"+str(x)+".jpg  -1\n" for x in easy_list[:size_mixture_set]]
        test_easy_mixture_file.writelines(img_set)
        img_set=["JPEGImages/"+str(x)+".jpg  -1\n" for x in hard_list[:size_mixture_set]]
        test_hard_mixture_file.writelines(img_set)
        test_easy_mixture_file.close()
        test_hard_mixture_file.close()
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
            test_single_file.writelines(img_set)
        train_file.close()
        val_file.close()
        test_single_file.close()
        test_easy_mixture_file.close()
        test_hard_mixture_file.close()
        with open("../data/osr_splits/"+name+"/split/" + str(name) + "_cls_info_split_" + str(split + 1) + ".json", "w") as outfile:
            json.dump(cls_info, outfile,cls=NpEncoder)


def VOC_osr_class_split(name="voc",root_path="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\"):
    print("Creating {0} OSR split".format(name))
    name_list = {}
    mixed_name_list=[]
    cls_label_dict=voc_cls_labels_dict
    img_name_list = [np.loadtxt(os.path.join(root_path, "voc_multi_train.txt"), dtype=float).astype(int),
                     np.loadtxt(os.path.join(root_path, "voc_multi_val.txt"), dtype=float).astype(int)]
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
        else:
            mixed_name_list.append(img)
    full_cls_set=list((nbr_cls>=50).nonzero()[0])
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])
    print("Would sample {0} training classes and {1} open classes".format(nbr_known_cls,len(full_cls_set)-nbr_known_cls))
    cls_info={}
    knwon_cls=list(full_train_cls)
    unknown_cls=list(set(full_cls_set).difference(set(knwon_cls)))
    cls_info["known_cls"]=knwon_cls
    cls_info["unknown_cls"]=unknown_cls
    print("train classes {0}, open classes {1} ".format(knwon_cls,unknown_cls))
    train_file=open("../data/osr_splits/" + name + "/"+str(name)+"_train_split.txt","w+")
    val_file=open("../data/osr_splits/" + name + "/"+str(name)+"_val_split.txt","w+")
    test_file = open("../data/osr_splits/" + name + "/"+str(name) + "_single_test_split.txt", "w+")
    test_easy_mixture_file = open(
        "../data/osr_splits/" + name + "/" + str(name) + "_easy_mixture_test_split.txt",
        "w+")
    test_hard_mixture_file = open(
        "../data/osr_splits/" + name + "/" + str(name) + "_hard_mixture_test_split.txt",
        "w+")
    easy_list=[]
    hard_list=[]
    for img in mixed_name_list:
        label = cls_label_dict[int(img)]
        if label[unknown_cls].sum() == 1 and label[knwon_cls].sum() == 1:  ## easy mixture
            easy_list.append(img)
        elif (label[unknown_cls].sum() >= 1) and label[unknown_cls].sum() < label[knwon_cls].sum():  ## hard mixture
            hard_list.append(img)
    if len(hard_list) < len(unknown_cls) * 50:
        size_mixture_set = len(hard_list)
    print(size_mixture_set)
    img_set = ["JPEGImages/" + str(x) + ".jpg  -1\n" for x in easy_list[:size_mixture_set]]
    test_easy_mixture_file.writelines(img_set)
    img_set = ["JPEGImages/" + str(x) + ".jpg  -1\n" for x in hard_list[:size_mixture_set]]
    test_hard_mixture_file.writelines(img_set)
    test_easy_mixture_file.close()
    test_hard_mixture_file.close()

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
    train_img_name_list = np.loadtxt(os.path.join(root_path, "coco_multi_train.txt"), dtype=str)
    val_img_name_list   = np.loadtxt(os.path.join(root_path, "coco_multi_val.txt"), dtype=str)
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
        #print(label.sum())
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
        train_file=open("../data/osr_splits/" + name + "/" +str(name)+"_train_split.txt","w+")
        val_file=open("../data/osr_splits/" + name + "/" +str(name)+"_val_split.txt","w+")
        test_file = open("../data/osr_splits/" + name + "/"  + str(name) + "_test_split.txt", "w+")

        test_easy_mixture_file = open(
            "../data/osr_splits/" + name + "/" + str(name) + "_easy_mixture_test_split.txt",
            "w+")
        test_hard_mixture_file = open(
            "../data/osr_splits/" + name + "/" + str(name) + "_hard_mixture_test_split.txt",
            "w+")

        easy_list = []
        hard_list = []
        for img in img_name_list:
            pre_fix = img.split('_')[1]
            key = int(img.split('_')[-1])
            label = cls_label_dict[int(key)]
            if label[unknown_cls].sum() == 1 and label[knwon_cls].sum() == 1:  ## easy mixture
                easy_list.append(pre_fix+'/'+img)
            elif (label[unknown_cls].sum() >= 1) and label[unknown_cls].sum() < label[
                    knwon_cls].sum():  ## hard mixture
                hard_list.append(pre_fix+'/'+img)
        if len(hard_list) < len(unknown_cls) * 50:
            size_mixture_set = len(hard_list)
        else:
            size_mixture_set=len(unknown_cls) * 50
        print(size_mixture_set)
        img_set = [str(x) + ".jpg  -1\n" for x in easy_list[:size_mixture_set]]
        test_easy_mixture_file.writelines(img_set)
        img_set = [ str(x) + ".jpg  -1\n" for x in hard_list[:size_mixture_set]]
        test_hard_mixture_file.writelines(img_set)
        test_easy_mixture_file.close()
        test_hard_mixture_file.close()


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
        knwon_cls.sort()
        unknown_cls=list(set(full_cls_set).difference(set(knwon_cls)))
        unknown_cls.sort()
        cls_info={}
        cls_info["known_cls"]=knwon_cls
        cls_info["unknown_cls"]=unknown_cls
        print("Split {0}: train classes {1}, open classes {2} ".format(split+1,knwon_cls,unknown_cls))
        train_file=open("../data/osr_splits/"+name+"/split/"+str(name)+"_train_split_"+str(split+1)+".txt","w+")
        val_file=open("../data/osr_splits/"+name+"/split/"+str(name)+"_val_split_"+str(split+1)+".txt","w+")
        test_file = open("../data/osr_splits/"+name+"/split/" + str(name) + "_test_split_" + str(split + 1) + ".txt", "w+")
        test_easy_mixture_file = open(
            "../data/osr_splits/" + name + "/split/" + str(name) + "_easy_mixture_test_split_" + str(split + 1) + ".txt",
            "w+")
        test_hard_mixture_file = open(
            "../data/osr_splits/" + name + "/split/" + str(name) + "_hard_mixture_test_split_" + str(split + 1) + ".txt",
            "w+")

        easy_list = []
        hard_list = []
        for img in img_name_list:
            pre_fix = img.split('_')[1]
            key = int(img.split('_')[-1])
            label = cls_label_dict[int(key)]
            if label[unknown_cls].sum() == 1 and label[knwon_cls].sum() == 1:  ## easy mixture
                easy_list.append(pre_fix+'/'+img)
            elif (label[unknown_cls].sum() >= 1) and label[unknown_cls].sum() < label[
                    knwon_cls].sum():  ## hard mixture
                hard_list.append(pre_fix+'/'+img)
        if len(hard_list) < len(unknown_cls) * 50:
            size_mixture_set = len(hard_list)
        print(size_mixture_set)
        img_set = [str(x) + ".jpg  -1\n" for x in easy_list[:size_mixture_set]]
        test_easy_mixture_file.writelines(img_set)
        img_set = [ str(x) + ".jpg  -1\n" for x in hard_list[:size_mixture_set]]
        test_hard_mixture_file.writelines(img_set)
        test_easy_mixture_file.close()
        test_hard_mixture_file.close()

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
        with open("../data/osr_splits/"+name+"/split/" + str(name) + "_cls_info_split_" + str(split + 1) + ".json", "w") as outfile:
            json.dump(cls_info, outfile,cls=NpEncoder)
if __name__=="__main__":
    #create_VOC_osr_class_split()
    VOC_osr_class_split()
    #create_COCO_osr_class_split()
    #COCO_osr_class_split()

