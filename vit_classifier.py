import torch
import timm
from torch import nn
import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision
import json
import torch.optim as optim
from utils.utils import get_available_devices
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from utils.evaluator import OSREvaluator
from utils import evaluation
from sklearn.metrics import average_precision_score
import numpy as np


def ood_eval(k_ood_score,u_ood_score):
    #if self.exp_type=="single":
    results = evaluation.metric_ood(k_ood_score, u_ood_score)['Bas']
    ap_score = average_precision_score([0] * len(k_ood_score) + [1] * len(u_ood_score),
                                        list(-k_ood_score) + list(-u_ood_score))
    results['AUPR'] = ap_score*100

    return results

class OSR_dataset(Dataset):
    def __init__(self, data_root="D:\\datasets\\VOC\\VOCdevkit\\",prefix='train',exp="coco", transform=None,indices=1):
        self.root_dir=data_root
        self.transform=transform
        self.prefix=prefix
        self.indices=indices
        self.exp=exp
        self.name_list=self.__get_name_list()
        self.transform=transform
        self.max_num_object=1

        if exp=="voc":
            self.num_known_classes=6
        elif exp=="coco":
            self.num_known_classes=20

    def __get_name_list(self):
        text_file=os.path.join("./data/osr_splits/"+self.exp+"/single/"+self.exp+"_"+self.prefix+".txt")
        print(text_file)
        line_list=open(text_file).readlines()
        print("Totally have {} samples in {} set.".format(len(line_list),self.prefix))
        img_path=[]
        labels=[]
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

        return image,label

def Get_OSR_Datasets(train_transform, test_transform,dataroot="/root/yinxu/Dataset/VOC/VOCdevkit/VOC2012",exp="coco",indices=1):

    train_dataset = OSR_dataset(prefix='train',data_root=dataroot,exp=exp,transform=train_transform,indices=indices)
    val_dataset = OSR_dataset(prefix='val',data_root=dataroot,exp=exp,transform=test_transform,indices=indices)
    no_mixture_dataset = OSR_dataset(prefix='no_mixture_test',data_root=dataroot,exp=exp,transform=test_transform,indices=indices)
    all_mixture_dataset= OSR_dataset(prefix='all_mixture_test',data_root=dataroot,exp=exp,transform=test_transform,indices=indices)


    openness_easy_mixture_unknown_dataset = OSR_dataset(prefix='openness_easy_test',data_root=dataroot,exp=exp,transform=test_transform,indices=indices)
    openness_hard_mixture_unknown_dataset = OSR_dataset(prefix='openness_hard_test',data_root=dataroot,exp=exp,transform=test_transform,indices=indices)

    dominance_easy_mixture_unknown_dataset = OSR_dataset(prefix='dominance_easy_test',data_root=dataroot,exp=exp,transform=test_transform,indices=indices)
    dominance_hard_mixture_unknown_dataset = OSR_dataset(prefix='dominance_hard_test',data_root=dataroot,exp=exp,transform=test_transform,indices=indices)


    print('Train: ', len(train_dataset), 'Test: ', len(val_dataset), 'No_mixture:',
          len(no_mixture_dataset),'All_mixture:', len(all_mixture_dataset), 'Openness_easy_mixture:', len(openness_easy_mixture_unknown_dataset),'Openness_hard_mixture:', len(openness_hard_mixture_unknown_dataset),'Dominance_easy_mixture:', len(dominance_easy_mixture_unknown_dataset),'Dominance_hard_mixture:', len(dominance_hard_mixture_unknown_dataset))
    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'No_mixture': no_mixture_dataset,
        'all_mixture': all_mixture_dataset,
        'openness_easy_mixture_test':openness_easy_mixture_unknown_dataset,
        'openness_hard_mixture_test': openness_hard_mixture_unknown_dataset,
        'dominance_easy_mixture_test':dominance_easy_mixture_unknown_dataset,
        'dominance_hard_mixture_test': dominance_hard_mixture_unknown_dataset
    }

    return all_datasets


def OSR_fit(model,train_loader,val_loader,test_loader_list,nbr_cls,device):
    train_epoch_size = len(train_loader)
    optimizer = optim.Adam(model.head.parameters(), lr=0.01)

    current_iter = 0
    epoch=0
    max_steps=2000    
    lr_scheduler=CosineAnnealingLR(optimizer,T_max=max_steps)

    while current_iter<max_steps:
        model
        model.train()
        total_loss = 0
        correct, total=0,0
        for batch_idx, (input,labels) in enumerate(train_loader, 0):

            optimizer.zero_grad()
            input=input.to(device)
            labels=labels.to(device)
            outputs = model(input)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            prediction=outputs.data.max(1)[1].to(device).cpu().numpy()
            label=labels.cpu().numpy()
            total += labels.shape[0]
            correct += (prediction == label).sum()
            current_iter+=1
        epoch+=1
        total_loss /= train_epoch_size
        acc = float(correct) * 100. / float(total)
        print("Train | Epoch: {}, Loss: {:5f},  Acc: {:5f}, LR: {}".format(
            epoch, total_loss,acc,lr_scheduler.get_last_lr()))
        

    with torch.no_grad():
        known_test=[]
        for batch_idx, (input,labels) in enumerate(valloader, 0):
            input=input.to(device)
            labels=labels.to(device)
            outputs = model(input)
            known_test.append(outputs)
        known_test=torch.concatenate(known_test,dim=0).cpu().numpy()
        known_test=np.max(known_test,-1)
        for type, loader in test_loader_list.items():
            print(type)
            unknown_test=[]
            for batch_idx, (input,labels) in enumerate(loader, 0):
                input=input.to(device)
                labels=labels.to(device)
                outputs = model(input)
                unknown_test.append(outputs)
            unknown_test=torch.concatenate(unknown_test,dim=0).cpu().numpy()
            unknown_test=np.max(unknown_test,-1)
            ood_evaluations=ood_eval(known_test,unknown_test)
            print("Metrics",{
            "OOD": ood_evaluations
        })



if __name__=="__main__":
    device, available_gpus=get_available_devices()
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
    #all_densenet_models = timm.list_models('*dino*')
    #print(all_densenet_models)


    dataset_dict={
        "coco":{
            "path": '/root/yinxu/Dataset/coco/2014',
            "number":20
        },
        "voc":
        {
            "path":'/root/yinxu/Dataset/VOC/VOCdevkit/VOC2012',
            "number": 6
    }}

    for name,ds in dataset_dict.items():
        model = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=True,num_classes=ds['number'])
        #model.head = nn.Linear(model.head.in_features, )
        model.to(device)

        for i in range(1,2):
            datasets=Get_OSR_Datasets(train_transform,test_transform,exp=name,dataroot=ds['path'],indices=i)
            dataloaders = {}
            for k, v, in datasets.items():
                if k=='train':
                    shuffle = True
                    batch_size = 64
                else:
                    batch_size=64
                    shuffle=False
                dataloaders[k] = DataLoader(v, batch_size=batch_size,
                                            shuffle=shuffle, sampler=None, num_workers=0)

            trainloader = dataloaders['train']
            valloader = dataloaders['val']
            dataloaders.pop('train')
            dataloaders.pop('val')
            OSR_fit(model,trainloader,valloader,dataloaders,5,device=device)