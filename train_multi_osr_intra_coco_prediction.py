import os
import argparse

from data.open_set_datasets import get_multi_ood_datasets
import torch.optim as optim
import torch
import numpy as np
import ocl
import hydra
from sklearn import metrics
from omegaconf import DictConfig, OmegaConf
import hydra_zen
import omegaconf
from torchvision.utils import make_grid
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from data.osr_natural import Get_OSR_Datasets
from data.open_set_datasets import get_intra_coco_datasets
from models.openslot import Net
import math
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from utils.evaluator import OSREvaluator
from utils import transform
from utils.utils import get_available_devices,log_visualizations,multi_correct_slot
import warnings
from torch.optim.lr_scheduler import StepLR
from datetime import datetime




def OSR_fit(config,model,train_loader,val_loader,test_loader,visualizer,device):

    exp_path=os.path.join(config.trainer.log_root_path,config.trainer.exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    log_path=os.path.join(exp_path,datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    os.mkdir(log_path)
    nbr_cls=config.models.classifier.nbr_clses
    print(nbr_cls)
    writer = SummaryWriter(log_path)
    params = model.get_trainable_params_groups(different_lr=False)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    hyperarameter=config.optimizers
    if len(params) > 1:
        params[0]['lr'] = 0.01
    print(hyperarameter)
    train_epoch_size = len(train_loader)
    optimizer = optim.Adam(params, lr=hyperarameter["lr_rate"])
    lr_scheduler=StepLR(optimizer,step_size=hyperarameter["decay_epoch"]*train_epoch_size,gamma=0.5)
    evaluator=OSREvaluator(train_loader=train_loader,visualizer=visualizer,num_known_classes=nbr_cls,exp_type='multi')
    #print(config.max_steps//train_epoch_size)
    current_iter = 0
    epoch=0

    # weight=torch.load(config.Checkpoint_weight)
    # model.load_state_dict(weight,strict=True)
    # overall_test(evaluator,model, val_loader, test_loader, epoch, writer)
    # print(auroc)
    # print("The best accuray : {:5f}".format(auroc))
    # print("The best method : "+method+"")
    # print(parameter)


    # best_auroc=0.
    # best_parameter={}
    trainer_config=config.trainer
    max_steps=trainer_config.max_epoch*train_epoch_size
    while current_iter<max_steps:
        model.train()
        total_loss = 0
        fg_loss=0
        bg_loss=0
        gts = []
        preds = []
        for batch_idx, sample in enumerate(train_loader, 0):
            current_iter += 1
            optimizer.zero_grad()
            for key, values in sample.items():
                sample[key]=values.cuda()
            outputs=model(sample)
            
            slots,logits,fg_matching_loss,bg_cls_loss= \
                outputs["slots"],outputs["fg_pred"],outputs["fg_matching_loss"],outputs["bg_loss"]
            
            loss=fg_matching_loss+0.01*bg_cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            slot_pred=multi_correct_slot(logits)
            gt=sample['label'].to(device).cpu().numpy()
            slot_pred=slot_pred.cpu().numpy()
            gts.append(gt)
            preds.append(slot_pred)
            total_loss += loss
            fg_loss+=fg_matching_loss.item()
            bg_loss+=bg_cls_loss.item()
            lr_scheduler.step()
            if batch_idx < train_epoch_size - 1 and current_iter<max_steps:
                del slots, logits,
            if current_iter>=max_steps:
                break
        image=sample['img'].cuda()
        decoder_output = model.get_slot_attention_mask(image)
        log_visualizations(visualizer,writer,decoder_output,image,current_iter)
        epoch+=1
        total_loss /= train_epoch_size
        fg_loss /= train_epoch_size
        bg_loss /= train_epoch_size
        FinalMAPs = []
        gts=np.concatenate(gts, 0)
        preds=np.concatenate(preds, 0)
        for i in range(0, nbr_cls):
            precision, recall, thresholds = metrics.precision_recall_curve(gts[:,i], preds[:,i])
            FinalMAPs.append(metrics.auc(recall, precision))
        print("Train | Epoch: {}, Loss: {:5f}, Fg_loss: {:5f}, Bg_loss: {:5f}, mAP: {:5f}, LR: {}".format(epoch, total_loss,fg_loss, bg_loss, np.mean(FinalMAPs),lr_scheduler.get_last_lr()))
        writer.add_scalar('train_loss', total_loss, current_iter)
        # if epoch>=150 and (epoch%trainer_config.save_every_n_epoch)==0:
        #     overall_test(evaluator,model, val_loader, test_loader, epoch, writer)
    save_path=os.path.join(exp_path,trainer_config.exp_name + "_osr_epoch_"+str(epoch)+".pth")
    torch.save(model.state_dict(), save_path)   

    #overall_test(evaluator,model, val_loader, test_loader, epoch, writer)

def overall_test(evaluator,model, val_loader, test_loader_list, epoch, writer):
    evaluator.eval(model, val_loader, test_loader_list, epoch, writer, compute_acc=False,processor=["slotenergy","slotmax"])
    #auroc_max,parameter_max=evaluator.eval(model, val_loader, test_loader_list, epoch, writer, compute_acc=False,processor="slotmax")
    #evaluator.Odin_score_eval(model, val_loader, test_loader_list)
    #evaluator.Mahalanobis_score_eval(model,val_loader, test_loader_list)
    #return auroc,parameter,method

def main_worker(config):
    device, available_gpus=get_available_devices()
    #train_classes, open_set_classes = get_class_splits("voc")
    model = Net(config.models, checkpoint_path=config.Discovery_weights).to(device)
    #model.load_state_dict(torch.load("./checkpoints/multi_voc_osr.pth"))
    visualization=hydra_zen.instantiate(config.visualizations, _convert_="all")

    datasets = get_intra_coco_datasets(task_name=config.dataset.name)
    dataloaders = {}
    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        batch_size = 256 if "test" in k  else 64
        dataloaders[k] = DataLoader(v, batch_size=batch_size,
                                    shuffle=shuffle, sampler=None, num_workers=0)

    trainloader = dataloaders['train']
    valloader = dataloaders['val']
    dataloaders.pop('train')
    dataloaders.pop('val')
    #testloader = dataloaders['test']


    OSR_fit(config,model,trainloader,valloader,dataloaders,visualization,device)
if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description="path of the config file")
    parser.add_argument('--config_path', type=str, default="configs/multi_osr_intra_coco_config.yaml")
    args = parser.parse_args()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    cfg = OmegaConf.load(args.config_path)
    main_worker(cfg)
