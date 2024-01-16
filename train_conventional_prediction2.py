import os
import argparse
from ocl.decoding import OSR_classifier
from data.open_set_datasets import get_datasets,get_class_splits
from data.osr_natural import Get_Tiny_Dataset
import torch.optim as optim
import torch
import ocl
import hydra
from omegaconf import DictConfig, OmegaConf
import hydra_zen
import omegaconf
from torchvision.utils import make_grid
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from data.osr_natural import Get_OSR_Datasets
from models.openslot import Net
import math
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from utils.evaluator import OSREvaluator
from utils import transform
from utils.utils import get_available_devices,get_highest_slot,log_visualizations
import warnings
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

def load_checkpoint(path,model,optimizer):
    weight=torch.load(path)['model_state_dict']
    model.load_state_dict(weight)
    optimizer=torch.load(path)['optimizer_state_dict']
    epoch=torch.load(path)['epoch']
    return epoch,model,optimizer



def OSR_fit(config,model,train_loader,val_loader,test_loader_list,nbr_cls,visualizer,device):
    trainer_config=config.trainer
    exp_path=os.path.join(trainer_config.log_root_path,trainer_config.exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    log_path=os.path.join(exp_path,datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    #print(log_path)
    os.mkdir(log_path)
    writer = SummaryWriter(log_path)
    params = model.get_trainable_params_groups(different_lr=False)
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    hyperarameter=config.optimizers
    if len(params) > 1:
        params[0]['lr'] = 0.01
    print(hyperarameter)
    train_epoch_size = len(train_loader)
    optimizer = optim.Adam(params, lr=hyperarameter["lr_rate"])
    lr_scheduler=StepLR(optimizer,step_size=hyperarameter["decay_epoch"]*train_epoch_size,gamma=0.5)
    evaluator=OSREvaluator(train_loader=train_loader,visualizer=visualizer,num_known_classes=nbr_cls,exp_type="single")

    current_iter = 0
    epoch=0
    max_steps=trainer_config.max_epoch*train_epoch_size
    #epoch,optimizer,model=load_checkpoint("./voc_single_epoch50_osr.pth",model,optimizer)
    # path= "./log/single_tinyimagenet/single_tinyimagenet_osr.pth"
    # weight=torch.load(path)
    # model.load_state_dict(weight,strict=False)
    #path= "./single_voc_osr_bg.pth"
    # weight=torch.load(path)
    # from collections import OrderedDict
    # classifier=OrderedDict()
    # for key in list(weight.keys()):
    #     if 'aux_classifier.' in key:
    #         classifier[key[15:]]=weight[key]
    #
    # msg=model.aux_classifier.load_state_dict(classifier,strict=True)

    #evaluator.score_comparison(model, val_loader, test_loader_list)


    #overall_test(evaluator, model, val_loader, test_loader_list, epoch, writer)
    while current_iter<max_steps:
        model.train()
        total_loss = 0
        fg_loss=0
        bg_loss=0
        correct, total=0,0
        for batch_idx, sample in enumerate(train_loader, 0):
            current_iter += 1
            for key, values in sample.items():
                sample[key]=values.cuda()

            optimizer.zero_grad()
            outputs = model(sample)

            slots,logits,fg_matching_loss,bg_cls_loss,valid_slots = \
                outputs["slots"],outputs["fg_pred"],outputs["fg_matching_loss"],outputs["bg_loss"],outputs["valid_slots"]
            #slots.retain_grad()


            loss=fg_matching_loss+0.1*bg_cls_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            logits=get_highest_slot(logits,trainer_config.softmax_eval)
            prediction=logits.data.max(1)[1].to(device).cpu().numpy()
            label=sample['label'].to(device).cpu().numpy()
            total += label.shape[0]
            correct += (prediction == label).sum()

            total_loss += loss.item()
            fg_loss+=fg_matching_loss.item()
            bg_loss+=bg_cls_loss.item()
            if batch_idx < train_epoch_size - 1 and current_iter<max_steps:
                del slots, prediction,
            if current_iter>=max_steps:
                break
        if (epoch%trainer_config.log_every_n_epoch)==0:
            image=sample['img'].cuda()
            decoder_output = model.get_slot_attention_mask(image)
            log_visualizations(visualizer,writer,decoder_output,image,current_iter)
        epoch+=1
        total_loss /= train_epoch_size
        fg_loss /=train_epoch_size
        bg_loss/=train_epoch_size
        acc = float(correct) * 100. / float(total)
        print("Train | Epoch: {}, Loss: {:5f}, Fg_loss: {:5f}, Bg_loss: {:5f}, Acc: {:5f}, LR: {}".format(
            epoch, total_loss,fg_loss,bg_loss,acc,lr_scheduler.get_last_lr()))
        writer.add_scalar('train_loss', total_loss, current_iter)
        # if epoch>=10 and epoch%5==0:
        #     overall_test(evaluator, model, val_loader, test_loader_list, epoch, writer)
    save_path=os.path.join(exp_path,trainer_config.exp_name + "_osr.pth")
    torch.save(model.state_dict(), save_path)
    overall_test(evaluator, model, val_loader, test_loader_list, epoch, writer)


def overall_test(evaluator,model, val_loader, test_loader_list, epoch, writer):
    evaluator.eval(model, val_loader, test_loader_list, epoch, writer, processor=["slotenergy","slotmax"],compute_acc=True,oscr=True)
    #evaluator.Odin_score_eval(model, val_loader, test_loader_list)
    #evaluator.Mahalanobis_score_eval(model,val_loader, test_loader_list)

def main_worker(config):
    device, available_gpus=get_available_devices()
    train_classes, open_set_classes = get_class_splits(config.dataset.name)
    model = Net(config.models, checkpoint_path=config.Discovery_weights).to(device)
    visualization=hydra_zen.instantiate(config.visualizations, _convert_="all")

    datasets = Get_Tiny_Dataset(split_idx=4)
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
    valloader = dataloaders['test_known']
    dataloaders.pop('train')
    dataloaders.pop('test_known')

    OSR_fit(config,model,trainloader,valloader,dataloaders,len(train_classes),visualization,device)

if __name__ == "__main__":
    import os
    import torch
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    cfg = OmegaConf.load("configs/single_conventional_classification_config1.yaml")
    main_worker(cfg)
