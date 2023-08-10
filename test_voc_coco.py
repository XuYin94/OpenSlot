import argparse
import os
import open_world_cifar as datasets
import argparse
from tqdm import tqdm
import time
from datetime import datetime
import torch.optim as optim
import math
from torchvision import transforms
from models.slotattention import SlotAttentionAutoEncoder
import torch
from torchvision.utils import make_grid
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from models.model_discovery_voc import Net
from torch.utils.data import Dataset,DataLoader
from data import Voc,coco
from utils import transform

denormalize = transform.DeNormalize(std=[0.229, 0.224, 0.225],
                                               mean=[0.485, 0.456, 0.406])
def visualize(img,slots):
    visz_list=[]
    num_slot=slots.shape[0]
    img=denormalize(img)
    visz_list.append(img)
    for i in range(num_slot):
        attention=slots[i].unsqueeze(0).repeat(3,1,1)
        visz_list.append(attention*img)
    visz_list=torch.stack(visz_list,0)
    visz_list=make_grid(visz_list.cpu(),nrow=len(visz_list))
    return visz_list


def get_available_devices():
    sys_gpu = torch.cuda.device_count()

    device = torch.device('cuda:0' if sys_gpu > 0 else 'cpu')
    available_gpus = list(range(sys_gpu))
    return device, available_gpus

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--model_dir', default='./tmp/model10.ckpt', type=str, help='where to save models')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--image_size', default=224, type=tuple)
parser.add_argument('--num_slots', default=6, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=256, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--train_steps', default=250000, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')
parser.add_argument('--log_path', default='./log/test', type=str, help='number of workers for loading data')
parser.add_argument('--different_lr', default=False, type=bool, help='number of workers for loading data')
args = parser.parse_args()



trans = transform.Compose([transform.HorizontalFilp(),
                           transform.Crop(base_size=256, crop_height=224, crop_width=224, type='center'),
                           transform.ToTensor(),
                           transform.Normalize(std=[0.229, 0.224, 0.225],
                                               mean=[0.485, 0.456, 0.406])
                           ])

train_split = 'train_aug'
val_split='val_aug'
data_root = "D:\\datasets\\VOC\\VOCdevkit\\VOC2012"

voc_set = Voc.VOC_Dataset(split=val_split, data_root=data_root, transform=trans)
voc_dataloader = DataLoader(voc_set, batch_size=2, shuffle=True, num_workers=0)
data_root = "D:\\datasets\\coco\\2014\\"
data_list = "D:\\datasets\\coco\\2014\\coco_multi_train.txt"
gt_mask = "D:\\datasets\\coco\\2014\\mask\\"
COCO_set = coco.COCO14SegmentationDataset(img_name_list_path=data_list, label_dir=gt_mask, coco14_root=data_root,
                                     transform=trans)
col_map = COCO_set.palette
coco_dataloader = DataLoader(COCO_set, batch_size=2, shuffle=True, num_workers=0)
device, available_gpus=get_available_devices()
model=Net(image_size=args.image_size,num_slot=args.num_slots,num_iteration=args.num_iterations,
          slot_dim=args.hid_dim).to(device)

new_model=Net(image_size=args.image_size,num_slot=args.num_slots+1,num_iteration=args.num_iterations,
          slot_dim=args.hid_dim).to(device)

if len(available_gpus)>0:
    model = torch.nn.DataParallel(model, device_ids=available_gpus)
    new_model = torch.nn.DataParallel(new_model, device_ids=available_gpus)

dict=torch.load("./checkpoints/oroc_pascal_voc_best_model.pth")
model.load_state_dict(dict,strict=True)
msg=new_model.load_state_dict(torch.load("./checkpoints/oroc_pascal_voc_best_model.pth"),strict=True)
#print(msg)
log_dir = os.path.join(args.log_path)
writer = SummaryWriter(log_dir)
model.eval()
new_model.eval()
voc_evaluation=[]
for batch_idx, sample in enumerate(voc_dataloader):
    image = sample['img']
    image = image.to(device)
    slots, soft_mask, __= model(image)
    __, __, height, width = image.shape
    soft_mask = torch.nn.functional.interpolate(soft_mask, (height, width), mode='bilinear')
    val_visz_list = visualize(image[-1].cpu(), soft_mask[-1].cpu())
    if len(voc_evaluation)>=20:
        break
    voc_evaluation.append(val_visz_list)
voc_evaluation=torch.cat(voc_evaluation,dim=1)
writer.add_image('voc/original',voc_evaluation)


coco_evaluation = []
for batch_idx, sample in enumerate(coco_dataloader):
    image = sample['img']
    image = image.to(device)
    slots, soft_mask, __= new_model(image)

    __, __, height, width = image.shape
    soft_mask = torch.nn.functional.interpolate(soft_mask, (height, width), mode='bilinear')
    val_visz_list = visualize(image[-1].cpu(), soft_mask[-1].cpu())

    if len(coco_evaluation)>=20:
        break
    coco_evaluation.append(val_visz_list)

coco_evaluation=torch.cat(coco_evaluation,dim=1)
writer.add_image('coco/original',coco_evaluation)


coco_evaluation = []
for batch_idx, sample in enumerate(coco_dataloader):
    image = sample['img']
    image = image.to(device)
    slots, soft_mask, __= new_model(image)

    __, __, height, width = image.shape
    soft_mask = torch.nn.functional.interpolate(soft_mask, (height, width), mode='bilinear')
    val_visz_list = visualize(image[-1].cpu(), soft_mask[-1].cpu())

    if len(coco_evaluation)>=20:
        break
    coco_evaluation.append(val_visz_list)

coco_evaluation=torch.cat(coco_evaluation,dim=1)
writer.add_image('coco/original',coco_evaluation)