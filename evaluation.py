import os
import argparse

from data.datasets import get_mixed_osr_dataset_funcs
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


def Evaluation(config,model,train_loader,postive,negative_list,nbr_cls,visualizer,device):

    exp_path=os.path.join(config.trainer.log_root_path,config.trainer.exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    log_path=os.path.join(exp_path,datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    os.mkdir(log_path)
    writer = SummaryWriter(log_path)
    evaluator=OSREvaluator(train_loader=train_loader,visualizer=visualizer,writer=writer,num_known_classes=nbr_cls,exp_type='multi')

    if not config.Checkpoint_weight:
        weight=torch.load(config.Checkpoint_weights)
        model.load_state_dict(weight,strict=False)
    model.eval()
    evaluator.eval(model, postive, negative_list, compute_acc=False,processor=["slotenergy","slotmax"])

def main_worker(config):
    device, available_gpus=get_available_devices()
    #train_classes, open_set_classes = get_class_splits("voc")
    model = Net(config.models, checkpoint_path=config.Discovery_weights).to(device)
    #model.load_state_dict(torch.load("./checkpoints/multi_voc_osr.pth"))
    visualization=hydra_zen.instantiate(config.visualizations, _convert_="all")

    datasets = get_mixed_osr_dataset_funcs(config.dataset.name)
    dataloaders = {}
    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(v, batch_size=4,
                                    shuffle=shuffle, sampler=None, num_workers=config.dataset.num_workers)

    train_loader,positive= dataloaders.pop('train'),dataloaders.pop('postives')



    Evaluation(config,model,train_loader,positive,dataloaders,config.models.classifier.nbr_clses,visualization,device)
if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    cfg = OmegaConf.load("configs/multi_osr_voc_config.yaml")
    import torch
    main_worker(cfg)
