import os
import argparse

from data.open_set_datasets import get_datasets,get_class_splits
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
from data.osr_natural import Get_OSR_Datasets
from models.openslot import Net
import math
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from data import Voc
from utils.evaluator import OSREvaluator
from utils import transform
import warnings
from ocl.optimization import OptimizationWrapper
from datetime import datetime
from ocl.utils.trees import walk_tree_with_paths
from ocl.visualization_types import Visualization

def log_visualizations(visualzer,logger_experiment,outputs,images,global_step, phase="train"):

    visualizations = {}
    for name, vis in visualzer.items():
        if isinstance(vis,ocl.visualizations.Image):
            visualizations[name] = vis(images)
        elif isinstance(vis,ocl.visualizations.Mask):
            visualizations[name] = vis(mask=outputs.masks_as_image)
        elif isinstance(vis,ocl.visualizations.Segmentation):
            visualizations[name] = vis(image=images,mask=outputs.masks_as_image)
        else:
            NotImplementedError


    visualization_iterator = walk_tree_with_paths(
        visualizations, path=None, instance_check=lambda t: isinstance(t, Visualization)
    )
    for path, vis in visualization_iterator:
        try:
            str_path = ".".join(path)
            vis.add_to_experiment(
                experiment=logger_experiment,
                tag=f"{phase}/{str_path}",
                global_step=global_step,
            )
        except AttributeError:
            # The logger does not support the right data format.
            pass

def get_available_devices():
    sys_gpu = torch.cuda.device_count()

    device = torch.device('cuda:0' if sys_gpu > 0 else 'cpu')
    available_gpus = list(range(sys_gpu))
    return device, available_gpus

def OSR_fit(config,model,optimizer,train_loader,val_loader,test_loader,nbr_cls,visualizer,device):

    exp_path=os.path.join(config.log_root_path,config.exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    log_path=os.path.join(exp_path,datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    os.mkdir(log_path)
    writer = SummaryWriter(log_path)

    #evaluator=OSREvaluator(num_known_classes=nbr_cls)
    train_epoch_size = len(train_loader)
    current_iter = 0
    epoch=0
    while current_iter<config.max_steps:
        model.train()
        total_loss = 0
        correct, total=0,0
        #evaluator.eval(model,val_loader,test_loader,epoch,writer)
        for batch_idx, sample in enumerate(train_loader, 0):
            current_iter += 1
            if not current_iter%2000:
                learning_rate=optimizer.param_groups[1]['lr']/2

                optimizer.param_groups[0]['lr'] = learning_rate/10
                optimizer.param_groups[1]['lr'] = learning_rate
            image = sample['img'].to(device)
            label = sample['label'].to(device)
            class_label=sample['class_label'].to(device)

            optimizer.zero_grad()
            slots, pred, matching_loss = model(image, class_label)##mlp_pred: [b,num_slot,num_class]
            if config.softmax_eval:
                pred=torch.softmax(pred,dim=-1)
            pred=torch.argmax(pred.flatten(1,2),dim=-1)
            pred%=nbr_cls
            total += label.size(0)
            correct += (pred == label.data).sum()

            total_loss += matching_loss.item()
            matching_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if batch_idx < train_epoch_size - 1 and current_iter<config.max_steps:
                del slots, pred,
            if current_iter>=config.max_steps:
                break
        decoder_output = model.get_slot_attention_mask(image)
        log_visualizations(visualizer,writer,decoder_output,image,current_iter)
        epoch+=1
        total_loss /= train_epoch_size
        acc = float(correct) * 100. / float(total)
        print("Train | Step: {}, Loss: {:5f}, Acc: {:5f}, LR_1: {:6f},LR_2: {:6f}".format(current_iter, total_loss,acc,optimizer.param_groups[0]["lr"],optimizer.param_groups[1]["lr"]))
        writer.add_scalar('train_loss', total_loss, current_iter)

    save_path=os.path.join(exp_path,config.exp_name + "_osr.pth")
    torch.save(model, save_path)


def main_worker(config):
    device, available_gpus=get_available_devices()
    train_classes, open_set_classes = get_class_splits("voc")
    model = Net(config.models).to(device)
    visualization=hydra_zen.instantiate(config.visualizations)

    params = model.get_trainable_params_groups(different_lr=True)
    if len(params) > 1:
        params[0]['lr'] = 0.001
    optimizer = optim.Adam(params, lr=0.01)
    datasets = get_datasets(name="voc")
    dataloaders = {}
    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(v, batch_size=64,
                                    shuffle=shuffle, sampler=None, num_workers=0)

    trainloader = dataloaders['train']
    valloader = dataloaders['val']
    testloader = dataloaders['test']


    OSR_fit(config.trainer,model,optimizer,trainloader,valloader,testloader,len(train_classes),visualization,device)
if __name__ == "__main__":
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    cfg = OmegaConf.load("./configs/classification_config.yaml")
    main_worker(cfg)
