from torch.utils.data import Dataset,DataLoader
import os
import torch
import numpy as np
from PIL import Image
import random
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

voc_cls_labels_dict = np.load('../data/cls_labels_voc.npy', allow_pickle=True).item()
coco_cls_labels_dict = np.load('../data/cls_labels_coco.npy', allow_pickle=True).item()


def create_VOC_osr_dominating_class_split(name="voc",root_path="/root/yinxu/Dataset/VOC/VOCdevkit/VOC2012/"):
    print("Creating {0} OSR split".format(name))
    name_list = {}
    mixed_name_list=[]
    cls_label_dict=voc_cls_labels_dict
    img_name_list = [np.loadtxt(os.path.join(root_path, "voc_multi_train.txt"), dtype=str).astype(int),
                     np.loadtxt(os.path.join(root_path, "voc_multi_val.txt"), dtype=str).astype(int)]
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
    full_cls_set=list(np.arange(20)) ## category list
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])## number of the object categories that size over 450
    #test_cls=set(list(np.arange(21))).difference(set(full_train_cls))
    print("Would sample {0} training classes and {1} open classes".format(nbr_known_cls,15))
    sampled_known_cls=random.sample(full_train_cls,5)
    for split in range(5):
        cls_info={}

        index=full_train_cls.index(sampled_known_cls[split])
        known_cls=full_train_cls[:index]+full_train_cls[index+1:]
        known_cls.sort()
        unknown_cls=list(set(full_cls_set).difference(set(known_cls)))
        unknown_cls.sort()
        cls_info["known_cls"]=known_cls
        cls_info["unknown_cls"]=unknown_cls
        print("Split {0}: train classes {1}, open classes {2} ".format(split+1,known_cls,unknown_cls))
        train_file=open("../data/osr_splits/"+name+"/single/split/"+str(name)+"_train_split_"+str(split+1)+".txt","w+")
        val_file=open("../data/osr_splits/"+name+"/single/split/"+str(name)+"_val_split_"+str(split+1)+".txt","w+")
        no_mixture_file = open("../data/osr_splits/"+name+"/single/split/" + str(name) + "_no_mixture_split_" + str(split + 1) + ".txt", "w+")
        opennnes_easy_mixture_file = open("../data/osr_splits/"+name+"/single/split/" + str(name) + "_openness_easy_mixture_test_split_" + str(split + 1) + ".txt", "w+")
        opennnes_hard_mixture_file = open("../data/osr_splits/"+name+"/single/split/" + str(name) + "_openness_hard_mixture_test_split_" + str(split + 1) + ".txt", "w+")
        dominance_easy_mixture_file = open("../data/osr_splits/"+name+"/single/split/" + str(name) + "_dominance_easy_mixture_test_split_" + str(split + 1) + ".txt", "w+")
        dominance_hard_mixture_file = open("../data/osr_splits/"+name+"/single/split/" + str(name) + "_dominance_hard_mixture_test_split_" + str(split + 1) + ".txt", "w+")
    
        
        no_mixture,o_easy,o_hard,d_easy,d_hard=[],[],[],[],[]
        import cv2
        for img in img_name_list:
            label=cls_label_dict[int(img)]
            img=str(img)
            sum_unknown_cls=label[unknown_cls].sum()
            sum_known_cls=label[known_cls].sum()
            if (sum_known_cls>0) and sum_unknown_cls>0: ## mixture case
                openness=sum_known_cls/sum_unknown_cls
                if openness<1:
                    o_easy.append(img)
                elif openness>1:
                    o_hard.append(img)
                seg_img = cv2.imread(root_path + 'SegmentationClassAug/' + str(img[:4] + "_" + img[4:]) + ".png", 0)
                seg_img = seg_img.flatten()
                unknown_pixel = 0
                known_pixel = 0
                for cls in full_cls_set:
                    nbr_pixel=np.where(seg_img==(cls+1),1,0).sum()
                    if cls in known_cls:
                        known_pixel+=nbr_pixel
                    else:
                        unknown_pixel += nbr_pixel
                if unknown_pixel>0 and known_pixel>0:
                    if (unknown_pixel>known_pixel) : ## easy mixture
                        d_easy.append(img)
                    else: ## hard mixture
                        d_hard.append(img)
            elif sum_known_cls==0 and sum_unknown_cls>0:
                no_mixture.append(img)
        test_size=min(len(no_mixture),len(o_easy),len(o_hard),len(d_easy),len(d_hard))
        file_list=[no_mixture,o_easy,o_hard,d_easy,d_hard]
        for idx,file in enumerate([no_mixture_file,opennnes_easy_mixture_file,opennnes_hard_mixture_file,dominance_easy_mixture_file,dominance_hard_mixture_file]):
            for i in range(test_size):
                img=file_list[idx][i]
                #print(name)
                file.writelines("JPEGImages/"+str(img)[:4]+"_"+str(img)[4:]+".jpg -1\n")

        for cls_index,train_cls in enumerate(known_cls):
            img_set=name_list[str(train_cls)][:nbr_img_per_cls]
            img_set=["JPEGImages/"+str(x)[:4]+"_"+str(x)[4:]+".jpg  "+str(cls_index)+"\n" for x in img_set]
            train_file.writelines(img_set)
            img_set=name_list[str(train_cls)][-50:]
            img_set=["JPEGImages/"+str(x)[:4]+"_"+str(x)[4:]+".jpg  "+str(cls_index)+"\n" for x in img_set]
            val_file.writelines(img_set)

        train_file.close()
        val_file.close()
        with open("../data/osr_splits/"+name+"/single/split/" + str(name) + "_cls_info_split_" + str(split + 1) + ".json", "w") as outfile:
            json.dump(cls_info, outfile,cls=NpEncoder)



def create_COCO_osr_class_split(name="coco",root_path="/root/yinxu/Dataset/coco/2014/"):
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

    full_cls_set=list(np.arange(80))
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])

    print("Would sample {0} training classes and {1} open classes".format(nbr_known_cls,64))
    for split in range(5):

        from random import sample
        known_cls=list(sample(full_train_cls,nbr_known_cls))
        known_cls.sort()
        unknown_cls=list(set(full_cls_set).difference(set(known_cls)))
        unknown_cls.sort()
        cls_info={}
        cls_info["known_cls"]=known_cls
        cls_info["unknown_cls"]=unknown_cls
        print("Split {0}: train classes {1}, open classes {2} ".format(split+1,known_cls,unknown_cls))
        train_file=open("../data/osr_splits/"+name+"/single/split/"+str(name)+"_train_split_"+str(split+1)+".txt","w+")
        val_file=open("../data/osr_splits/"+name+"/single/split/"+str(name)+"_val_split_"+str(split+1)+".txt","w+")

        no_mixture_file = open("../data/osr_splits/"+name+"/single/split/" + str(name) + "_no_mixture_split_" + str(split + 1) + ".txt", "w+")
        openness_easy_mixture_file = open(
            "../data/osr_splits/" + name + "/single/split/" + str(name) + "_openness_easy_mixture_test_split_" + str(split + 1) + ".txt",
            "w+")
        openness_hard_mixture_file = open(
            "../data/osr_splits/" + name + "/single/split/" + str(name) + "_openness_hard_mixture_test_split_" + str(split + 1) + ".txt",
            "w+")
        dominance_easy_mixture_file = open(
            "../data/osr_splits/" + name + "/single/split/" + str(name) + "_dominance_easy_mixture_test_split_" + str(split + 1) + ".txt",
            "w+")
        dominance_hard_mixture_file = open(
            "../data/osr_splits/" + name + "/single/split/" + str(name) + "_dominance_hard_mixture_test_split_" + str(split + 1) + ".txt",
            "w+")

        no_mixture,o_easy,o_hard,d_easy,d_hard=[],[],[],[],[]
        import cv2
        for img in img_name_list:
            img = str(img)
            pre_fix = img.split('_')[1]
            #print(pre_fix)
            key = int(img.split('_')[-1])
            label = cls_label_dict[int(key)]

            sum_unknown_cls=label[unknown_cls].sum()
            sum_known_cls=label[known_cls].sum()
            if (sum_known_cls>0) and sum_unknown_cls>0: ## mixture case
                openness=sum_known_cls/sum_unknown_cls
                if openness<1:
                    o_easy.append(pre_fix+'/'+img)
                elif openness>1:
                    o_hard.append(pre_fix+'/'+img)

                if "train" in pre_fix:
                    #print(root_path + 'mask/' + str(key) + ".png")
                    seg_img = cv2.imread(root_path + 'mask/' + str(key) + ".png", 0)
                else:
                    #print(root_path + 'mask/' + str(key) + ".png")
                    seg_img = cv2.imread(root_path + 'gt_val/' + str(key) + ".png", 0)
                seg_img = seg_img.flatten()
                #print(np.unique(seg_img))
                #assert np.max(seg_img)<=80
                unknown_pixel = 0
                known_pixel = 0
                for cls in full_cls_set:
                    nbr_pixel = np.where(seg_img == (cls + 1), 1, 0).sum()
                    if cls in known_cls:
                        known_pixel += nbr_pixel
                    else:
                        unknown_pixel += nbr_pixel
                if unknown_pixel > 0 and known_pixel > 0:
                    if (unknown_pixel > known_pixel):  ## easy mixture
                        d_easy.append(pre_fix+'/'+img)
                    else:  ## hard mixture
                        d_hard.append(pre_fix+'/'+img)
            elif sum_known_cls==0 and sum_unknown_cls>0:
                no_mixture.append(pre_fix+'/'+img)
        test_size=min(len(no_mixture),len(o_easy),len(o_hard),len(d_easy),len(d_hard))
        file_list=[no_mixture,o_easy,o_hard,d_easy,d_hard]
        for idx,file in enumerate([no_mixture_file,openness_easy_mixture_file,openness_hard_mixture_file,dominance_easy_mixture_file,dominance_hard_mixture_file]):
            for i in range(test_size):
                img=file_list[idx][i]
                #print(name)
                file.writelines(""+str(img)+".jpg -1\n")

        for cls_index,train_cls in enumerate(known_cls):
            img_set=name_list[str(train_cls)][:nbr_img_per_cls]
            img_set=[str(x)+" "+str(cls_index)+"\n" for x in img_set]
            train_file.writelines(img_set)
            img_set=name_list[str(train_cls)][-50:]
            img_set=[str(x)+" "+str(cls_index)+"\n" for x in img_set]
            val_file.writelines(img_set)
        train_file.close()
        val_file.close()
        with open("../data/osr_splits/"+name+"/single/split/" + str(name) + "_cls_info_split_" + str(split + 1) + ".json", "w") as outfile:
            json.dump(cls_info, outfile,cls=NpEncoder)




def VOC_single_label_osr_class_split(name="voc",root_path="/root/yinxu/Dataset/VOC/VOCdevkit/VOC2012/"):
    print("Creating VOC-6/14 OSR split")
    name_list = {}
    #mixed_name_list=[]
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
    test_imgs=[]
    for idx,img in enumerate(img_name_list):
        label=cls_label_dict[int(img)]
        if label.sum()==1:
            nbr_cls[label==1]+=1
            name_list[str((label==1).nonzero()[0][0])].append(img)
        else:
            test_imgs.append(img)
    full_cls_set=list(np.arange(20)) ## category list
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])## number of the object categories that size over 450
    print("Would sample {0} training classes and {1} open classes".format(nbr_known_cls,len(full_cls_set)-nbr_known_cls))
    cls_info={}
    knwon_cls=list(full_train_cls)
    unknown_cls=list(set(full_cls_set).difference(set(knwon_cls)))
    cls_info["known_cls"]=knwon_cls
    cls_info["unknown_cls"]=unknown_cls
    print("train classes: {0}, open classes: {1} ".format(knwon_cls,unknown_cls))
    train_file=open("../data/osr_splits/" + name + "/single/"+str(name)+"_train.txt","w+")
    known_test_file=open("../data/osr_splits/" + name + "/single/"+str(name)+"_val.txt","w+")
    no_mixture_test_file = open("../data/osr_splits/" + name + "/single/"+str(name) + "_no_mixture_test.txt", "w+")
    all_mixtur_test_file = open("../data/osr_splits/" + name + "/single/"+str(name) + "_all_mixture_test.txt", "w+")

    openness_easy_test_file = open(
        "../data/osr_splits/" + name + "/single/" + str(name) + "_openness_easy_test.txt",
        "w+")
    openness_hard_test_file = open(
        "../data/osr_splits/" + name + "/single/" + str(name) + "_openness_hard_test.txt",
        "w+")
    dominance_easy_test_file = open(
        "../data/osr_splits/" + name + "/single/" + str(name) + "_dominance_easy_test.txt",
        "w+")
    dominance_hard_test_file = open(
        "../data/osr_splits/" + name + "/single/" + str(name) + "_dominance_hard_test.txt",
        "w+")


    import cv2
    for img in test_imgs:
        img=str(img)
        seg_img=cv2.imread(root_path+'SegmentationClassAug/'+str(img[:4]+"_"+img[4:])+".png",0)
        seg_img=seg_img.flatten()
        unknown_pixel=0
        known_pixel=0

        label = cls_label_dict[int(img)]

        sum_unknown_cls=label[unknown_cls].sum()
        sum_known_cls=label[knwon_cls].sum()

        if (sum_known_cls>0) and sum_unknown_cls>0: ## mixture case

            all_mixtur_test_file.writelines("JPEGImages/"+str(img [:4]+"_"+img [4:])+".jpg  -1\n")


            openness=sum_known_cls/sum_unknown_cls
            if openness<1:
                openness_easy_test_file.writelines("JPEGImages/"+str(img [:4]+"_"+img [4:])+".jpg  -1\n")
            elif openness>1:
                openness_hard_test_file.writelines("JPEGImages/"+str(img [:4]+"_"+img [4:])+".jpg  -1\n")


            for cls in full_cls_set:
                nbr_pixel=np.where(seg_img==(cls+1),1,0).sum()
                if cls in knwon_cls:
                    known_pixel+=nbr_pixel
                else:
                    unknown_pixel += nbr_pixel
            if unknown_pixel>0 and known_pixel>0:
                if (unknown_pixel>known_pixel) : ## easy mixture
                    dominance_easy_test_file.writelines("JPEGImages/"+str(img [:4]+"_"+img [4:])+".jpg  -1\n")
                else: ## hard mixture
                    dominance_hard_test_file.writelines("JPEGImages/"+str(img [:4]+"_"+img [4:])+".jpg  -1\n")
        elif (sum_known_cls > 0) and sum_unknown_cls == 0:  ## only contains known classes
            known_test_file.writelines("JPEGImages/"+str(img [:4]+"_"+img [4:])+".jpg  20\n")
        
        elif (sum_known_cls==0) and (sum_unknown_cls>0): ## only contains unknown classes
            no_mixture_test_file.writelines("JPEGImages/"+str(img [:4]+"_"+img [4:])+".jpg  -1\n")

    
    dominance_easy_test_file.close()
    dominance_hard_test_file.close()
    openness_easy_test_file.close()
    openness_hard_test_file.close()
    no_mixture_test_file.close()

    
    all_mixtur_test_file.close()

    for cls_index,train_cls in enumerate(knwon_cls):
        img_set=name_list[str(train_cls)][:nbr_img_per_cls]
        img_set=["JPEGImages/"+str(x)[:4]+"_"+str(x)[4:]+".jpg  "+str(cls_index)+"\n" for x in img_set]
        train_file.writelines(img_set)

        # add (no-training) single-label images into the test set of KKCs
        img_set=name_list[str(train_cls)][nbr_img_per_cls:]
        img_set=["JPEGImages/"+str(x)[:4]+"_"+str(x)[4:]+".jpg  20\n" for x in img_set]
        known_test_file.writelines(img_set)

    known_test_file.close()
    train_file.close()
    #val_file.close()
    #no_mixture_test_file.close()
    with open("../data/osr_splits/" + name + "/single/" + str(name) + "_cls_info_split.json", "w") as outfile:
        json.dump(cls_info, outfile,cls=NpEncoder)


def COCO_single_label_osr_class_split(name="coco",root_path="/root/yinxu/Dataset/coco/2014/"):
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
    test_img_list=[]
    img_name_list=np.concatenate([train_img_name_list,val_img_name_list],axis=0)
    assert img_name_list.size==len(cls_label_dict)
    for idx,img in enumerate(img_name_list):
        pre_fix=img.split('_')[1]
        key=int(img.split('_')[-1])
        label=cls_label_dict[key]
        #print(label.sum())
        if label.sum()==1:
            nbr_cls[label==1]+=1
            name_list[str((label==1).nonzero()[0][0])].append(pre_fix+"/"+img+".jpg")
        else:
            test_img_list.append(img)
    full_cls_set=np.arange(80)
    full_train_cls=list((nbr_cls>=nbr_img_per_cls+50).nonzero()[0])

    print("Would sample {0} training classes and {1} open classes".format(nbr_known_cls,len(full_cls_set)-nbr_known_cls))

    from random import sample
    knwon_cls=full_train_cls
    unknown_cls=list(set(full_cls_set).difference(set(knwon_cls)))
    knwon_cls.sort()
    unknown_cls.sort()
    cls_info={}
    cls_info["known_cls"]=knwon_cls
    cls_info["unknown_cls"]=unknown_cls
    print("train classes {0}, open classes {1} ".format(knwon_cls,unknown_cls))
    train_file=open("../data/osr_splits/" + name + "/single/"+str(name)+"_train.txt","w+")
    known_test_file=open("../data/osr_splits/" + name + "/single/"+str(name)+"_val.txt","w+")
    no_mixture_test_file = open("../data/osr_splits/" + name + "/single/"+str(name) + "_no_mixture_test.txt", "w+")
    all_mixture_test_file = open("../data/osr_splits/" + name + "/single/"+str(name) + "_all_mixture_test.txt", "w+")

    openness_easy_test_file = open(
        "../data/osr_splits/" + name + "/single/" + str(name) + "_openness_easy_test.txt",
        "w+")
    openness_hard_test_file = open(
        "../data/osr_splits/" + name + "/single/" + str(name) + "_openness_hard_test.txt",
        "w+")
    dominance_easy_test_file = open(
        "../data/osr_splits/" + name + "/single/" + str(name) + "_dominance_easy_test.txt",
        "w+")
    dominance_hard_test_file = open(
        "../data/osr_splits/" + name + "/single/" + str(name) + "_dominance_hard_test.txt",
        "w+")


    import cv2
    for img in test_img_list:
        img = str(img)
        pre_fix = img.split('_')[1]
        key = int(img.split('_')[-1])
        label = cls_label_dict[int(key)]
        
        sum_unknown_cls=label[unknown_cls].sum()
        sum_known_cls=label[knwon_cls].sum()

        if (sum_known_cls>0) and sum_unknown_cls>0: ## mixture case

            all_mixture_test_file.writelines(pre_fix+"/"+img+".jpg  -1\n")
            openness=sum_known_cls/sum_unknown_cls
            if openness<1:
                openness_easy_test_file.writelines(pre_fix+"/"+img+".jpg  -1\n")
            elif openness>1:
                openness_hard_test_file.writelines(pre_fix+"/"+img+".jpg -1\n")

            if "train" in pre_fix:
                seg_img = cv2.imread(root_path + 'mask/' + str(key) + ".png", 0)
            else:
                #print(root_path + 'mask/' + str(key) + ".png")
                seg_img = cv2.imread(root_path + 'gt_val/' + str(key) + ".png", 0)
            seg_img = seg_img.flatten()
            #print(np.unique(seg_img))
            #assert np.max(seg_img)<=80
            unknown_pixel = 0
            known_pixel = 0
            for cls in full_cls_set:
                nbr_pixel = np.where(seg_img == (cls + 1), 1, 0).sum()
                if cls in knwon_cls:
                    known_pixel += nbr_pixel
                else:
                    unknown_pixel += nbr_pixel
            if unknown_pixel > 0 and known_pixel > 0:
                #print(img)
                if (unknown_pixel > known_pixel):  ## easy mixture
                    dominance_easy_test_file.writelines(pre_fix+"/"+img+".jpg  -1\n")
                else:  ## hard mixture
                    dominance_hard_test_file.writelines(pre_fix+"/"+img+".jpg  -1\n")

        elif (sum_known_cls > 0) and sum_unknown_cls == 0:  ## only contains known classes
            known_test_file.writelines(pre_fix + "/" + img + ".jpg   20\n")
        
        elif (sum_known_cls==0) and (sum_unknown_cls>0): ## only contains unknown classes
            no_mixture_test_file.writelines(pre_fix + "/" + img + ".jpg   -1\n")

        # elif (sum_known_cls == 0) and sum_unknown_cls >0:  ## mixture case
        #     no_mixture_test_file.writelines(pre_fix + "/" + img + ".jpg -1\n")

        # elif (sum_known_cls > 0) and sum_unknown_cls == 0:  ## mixture case
        #     no_mixture_test_file.writelines(pre_fix + "/" + img + ".jpg -1\n")

    for cls_index,train_cls in enumerate(knwon_cls):
        img_set=name_list[str(train_cls)][:nbr_img_per_cls]
        img_set=[str(x)+" "+str(cls_index)+"\n" for x in img_set]
        train_file.writelines(img_set)
        img_set=name_list[str(train_cls)][nbr_img_per_cls:]
        img_set=[str(x)+" "+str(cls_index)+"\n" for x in img_set]
        known_test_file.writelines(img_set)
    # for test_cls in unknown_cls:
    #     img_set=name_list[str(test_cls)][-50:]
    #     img_set=[str(x)+" -1\n" for x in img_set]
    #     no_mixture_test_file.writelines(img_set)

    train_file.close()
    #val_file.close()
    no_mixture_test_file.close()
    with open("../data/osr_splits/" + name + "/single/" + str(name) + "_cls_info_split.json", "w") as outfile:
        json.dump(cls_info, outfile,cls=NpEncoder)


voc2coco_map=[4,1,15,8,43,5,2,16,61,20,66,17,18,3,0,63,19,62,6,71]
root_path="/root/yinxu/Dataset/coco/2014/"


def get_non_overlap_coco_img():

    coco_label_dict=np.concatenate((np.load("./osr_splits/coco/multi/coco_multi_train_label.npy"),
                                       np.load("./osr_splits/coco/multi/coco_multi_val_label.npy")),0)

    #coco_label_dict=np.load("./osr_splits/coco/multi/coco_multi_val_label.npy")
    non_overplapped_cls=list(set(range(80)).difference(set(voc2coco_map)))


    img_list=open(r'./osr_splits/coco/multi/coco_multi_train.txt').readlines()\
             +open(r'./osr_splits/coco/multi/coco_multi_val.txt').readlines()
    #img_list=open(r'./osr_splits/coco/multi/coco_multi_val.txt').readlines()
    print(len(img_list))
    print(coco_label_dict.shape)
    no_mixture_file=open("osr_splits/multi_osr/voc2coco14/voc2coco14_osr_no_mixture.txt", "w+")
    all_mixture_file=open("osr_splits/multi_osr/voc2coco14/voc2coco14_osr_all_mixture.txt", "w+")
    openness_easy_file = open("osr_splits/multi_osr/voc2coco14/voc2coco14_osr_openness_easy.txt", "w+")
    openness_hard_file = open("osr_splits/multi_osr/voc2coco14/voc2coco14_osr_openness_hard.txt", "w+")
    dominance_easy_file = open("osr_splits/multi_osr/voc2coco14/voc2coco14_osr_dominance_easy.txt", "w+")
    dominance_hard_file = open("osr_splits/multi_osr/voc2coco14/voc2coco14_osr_dominance_hard.txt", "w+")




    for i in range(coco_label_dict.shape[0]):
        voc_cls_sum=coco_label_dict[i,voc2coco_map].sum()
        non_overlap_cls_sum=coco_label_dict[i,non_overplapped_cls].sum()
        name=img_list[i].strip()

        import cv2
        if non_overlap_cls_sum>0 and voc_cls_sum>0: ## contain at least one unknown class
            pre_fix = name.split('/')[1]
            key = int(name.split('/')[-1].split('_')[-1][:-4])

            all_mixture_file.writelines(name + "\n")
            wildness=voc_cls_sum/non_overlap_cls_sum
            if wildness<1:
                openness_easy_file.writelines(name + "\n")
            elif wildness>1:
                openness_hard_file.writelines(name+"\n")
            if "train" in pre_fix:
                #print(root_path + 'mask/' + str(key) + ".png")
                seg_img = cv2.imread(root_path + 'mask/' + str(key) + ".png", 0)
            else:
                # print(root_path + 'mask/' + str(key) + ".png")
                seg_img = cv2.imread(root_path + 'gt_val/' + str(key) + ".png", 0)
            if seg_img is None:
                continue
            unknown_pixel = 0
            known_pixel = 0
            for cls in range(80):
                nbr_pixel = np.where(seg_img == (cls + 1), 1, 0).sum()
                if cls in voc2coco_map:
                    known_pixel += nbr_pixel
                else:
                    unknown_pixel += nbr_pixel
            if unknown_pixel > 0 and known_pixel > 0:
                if (unknown_pixel > known_pixel):  ## easy mixture
                    dominance_easy_file.writelines(name+"\n")
                else:  ## hard mixture
                    dominance_hard_file.writelines(name+"\n")
            else:
                continue
        elif (voc_cls_sum==0) and (non_overlap_cls_sum>0): ## only contains unknown classes
            no_mixture_file.writelines(name+"\n")
            continue


def intra_coco_osr_class_split():
    nbr_known_classes=[40,60]
    coco_label_dict=np.concatenate((np.load("./osr_splits/coco/multi/coco_multi_train_label.npy"),
                                       np.load("./osr_splits/coco/multi/coco_multi_val_label.npy")),0)
    #coco_label_dict=np.load("./osr_splits/coco/multi/coco_multi_val_label.npy")
    coco_img_list=open(r'./osr_splits/coco/multi/coco_multi_train.txt').readlines()\
             +open(r'./osr_splits/coco/multi/coco_multi_val.txt').readlines()

    #coco_img_list=open(r'./osr_splits/coco/multi/coco_multi_val.txt').readlines()
    full_cls_set=np.arange(80)

    for i in range(len(nbr_known_classes)):
        from random import sample
        known_cls=list(sample(list(full_cls_set),nbr_known_classes[i])) ## sample certain number of classes as known for training
        unknown_cls=list(set(full_cls_set).difference(set(known_cls)))
        known_cls.sort()
        unknown_cls.sort()
        print(known_cls)
        cls_info={}
        cls_info["known_cls"]=known_cls
        cls_info["unknown_cls"]=unknown_cls
        train_file=open("/root/yinxu/open_set/test/Openslot/data/osr_splits/multi_osr/Intra_coco/Task" +str(i+2)+"/train_set.txt","w+")
        in_test_file=open("/root/yinxu/open_set/test/Openslot/data/osr_splits/multi_osr/Intra_coco/Task" +str(i+2)+"/in_test_set.txt","w+")
        test_file = open("/root/yinxu/open_set/test/Openslot/data/osr_splits/multi_osr/Intra_coco/Task" +str(i+2)+"/no_mixture_set.txt", "w+")
        all_mixture_file = open("/root/yinxu/open_set/test/Openslot/data/osr_splits/multi_osr/Intra_coco/Task" +str(i+2)+"/all_mixture_set.txt", "w+")
        openness_easy_file = open(
            "/root/yinxu/open_set/test/Openslot/data/osr_splits/multi_osr/Intra_coco/Task" +str(i+2)+"/openness_easy_test_set.txt",
            "w+")
        openness_hard_file = open(
            "/root/yinxu/open_set/test/Openslot/data/osr_splits/multi_osr/Intra_coco/Task" +str(i+2)+"/opennness_hard_test_set.txt",
            "w+")
        
        dominance_easy_file = open(
            "/root/yinxu/open_set/test/Openslot/data/osr_splits/multi_osr/Intra_coco/Task" +str(i+2)+"/dominance_easy_test_set.txt",
            "w+")
        dominance_hard_file = open(
            "/root/yinxu/open_set/test/Openslot/data/osr_splits/multi_osr/Intra_coco/Task" +str(i+2)+"/dominance_hard_test_set.txt",
            "w+")

        #train_list=[]
        label_list=[]

        import cv2
        for idx,img in enumerate(coco_img_list):
            img = str(img).strip()
            pre_fix = img.split('_')[1]
            #print(pre_fix)
            key = int(img.split('_')[-1][:-4])
            label = coco_label_dict[idx]
            
            sum_known_cls=label[known_cls].sum()
            sum_unknown_cls=label[unknown_cls].sum()

            if sum_known_cls>0 and sum_unknown_cls==0:  ## only contain known classes

                gt_label=torch.zeros(nbr_known_classes[i])
                #print((label!=0).nonzero()[0])
                original_cls_indices=(label!=0).nonzero()[0]
                #print(original_cls_indices)
                for j in original_cls_indices:
                    indices=(known_cls==j).nonzero()[0]
                    #print(indices)
                    gt_label[indices]=1
                    #print(indices)
                if "train" in pre_fix:
                    train_file.writelines(img+"\n")
                else:
                    in_test_file.writelines(img+"\n")  ## images from the validation set will be used for the in-distribution test 
                label_list.append(gt_label)
                continue

            elif sum_known_cls==0 and sum_unknown_cls>0:  ## only contain unknown classes (no_mixture)
                test_file.writelines(img+"-1\n")
                continue

            elif (sum_unknown_cls>0) and (sum_known_cls>0): ## mixture case
                all_mixture_file.writelines(img+"\n")
                openness=sum_known_cls/sum_unknown_cls
                if openness<1:
                    openness_easy_file.writelines(img+"\n")
                elif openness>1:
                    openness_hard_file.writelines(img+"\n")


                if "train" in pre_fix:
                    #print(root_path + 'mask/' + str(key) + ".png")
                    seg_img = cv2.imread(root_path + 'mask/' + str(key) + ".png", 0)
                else:
                    seg_img = cv2.imread(root_path + 'gt_val/' + str(key) + ".png", 0)
                if seg_img is None:
                    continue
                seg_img = seg_img.flatten()
                unknown_pixel = 0
                known_pixel = 0
                for cls in full_cls_set:
                    nbr_pixel = np.where(seg_img == (cls + 1), 1, 0).sum()
                    if cls in known_cls:
                        known_pixel += nbr_pixel
                    else:
                        unknown_pixel += nbr_pixel
                if unknown_pixel > 0 and known_pixel > 0:
                    if (unknown_pixel > known_pixel):  ## easy mixture
                        dominance_easy_file.writelines(img+"\n")
                    else:  ## hard mixture
                        dominance_hard_file.writelines(img+"\n")

        label_list=np.stack(label_list,axis=0)
        #print(label_list.shape)
        np.save("/root/yinxu/open_set/test/Openslot/data/osr_splits/multi_osr/Intra_coco/Task"+str(i+2)+"/train_label.npy",label_list)
        with open("/root/yinxu/open_set/test/Openslot/data/osr_splits/multi_osr/Intra_coco/Task"+str(i+2)+"/cls_info.json", "w") as outfile:
            json.dump(cls_info, outfile,cls=NpEncoder)


if __name__=="__main__":
    #create_VOC_osr_dominating_class_split()
    #create_COCO_osr_class_split()
    #VOC_osr_class_split()
    #COCO_osr_class_split()
    #get_non_overlap_coco_img()
    intra_coco_osr_class_split()

    #VOC_single_label_osr_class_split()
    #COCO_single_label_osr_class_split()