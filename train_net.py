import os
import open_world_cifar as datasets
import argparse
from utils import TransformTwice
from tqdm import tqdm
import time
from datetime import datetime
import torch.optim as optim
from models.slotattention import SlotAttentionAutoEncoder
import torch
from torchvision.utils import make_grid
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision


mean=[0.5071, 0.4867, 0.4408]
std=[0.2675, 0.2565, 0.2761]


def visualize(img,recon_combined, recons):
    # `recon_combined` has shape: [num_channels,width, height].
    # `masks` has shape: [num_slots, width, height].
    # `recons` has shape: [num_slots, num_channels, width, height].
    #num_slot,__,__,__=recons.shape
    recons=recons.permute(0,3,1,2)
    visz_list=[]
    num_slot=recons.shape[0]
    for t, m, s in zip(img,mean, std):
        t.mul_(s).add_(m)
    visz_list.append(img)
    for i in range(num_slot):
        slot=recons[i]
        for t, m, s in zip(slot, mean, std):
            t.mul_(s).add_(m)
        visz_list.append(slot)
    for t, m, s in zip(recon_combined,mean, std):
        t.mul_(s).add_(m)
    visz_list.append(recon_combined.squeeze(0))
    visz_list=make_grid(visz_list,nrow=len(visz_list))
    return visz_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--model_dir', default='./tmp/model10.ckpt', type=str, help='where to save models')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_slots', default=3, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=5, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')
parser.add_argument('--log_path', default='./log', type=str, help='number of workers for loading data')

opt = parser.parse_args()
resolution = (128, 128)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.CIFAR10(root='D:\\datasets\\', train=True,
                                            transform=TransformTwice(datasets.dict_transform['cifar_train']))

test_set = torchvision.datasets.CIFAR10(root='D:\\datasets\\', train=False,
                                     transform=datasets.dict_transform['cifar_test'])
num_classes = 10

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)

train_epoch_size = len(train_dataloader)
val_epoch_size = len(test_loader)

model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim).to(device)
# model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])

criterion = nn.MSELoss()

params = [{'params': model.parameters()}]

optimizer = optim.Adam(params, lr=opt.learning_rate)

log_dir = os.path.join(opt.log_path)
writer = SummaryWriter(log_dir)


start = time.time()
i = 0
for epoch in range(opt.num_epochs):
    model.train()

    total_loss = 0

    for batch_idx,data in enumerate(train_dataloader,0):
        i += 1
        if i < opt.warmup_steps:
            learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        else:
            learning_rate = opt.learning_rate

        learning_rate = learning_rate * (opt.decay_rate ** (
                i / opt.decay_steps))

        optimizer.param_groups[0]['lr'] = learning_rate

        img,__=data
        image = img[0].to(device)
        image=torch.nn.functional.interpolate(image, (128,128),mode='bilinear')
        recon_combined, recons, masks, slots = model(image)
        loss = criterion(recon_combined, image)
        total_loss += loss.item()

        if batch_idx<train_epoch_size-1:
            del recons, masks, slots

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i > opt.warmup_steps and i % 50 == 0:
            writer.add_scalar('train_loss', total_loss, i)

    total_loss /= len(train_dataloader)

    print("Train | Epoch: {}, Loss: {}".format(epoch, total_loss))

    with torch.no_grad():
        train_visz_list=visualize(image[-1],recon_combined[-1],recons[-1])
        writer.add_image("train_slot",train_visz_list,epoch)


    if not epoch % 10:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, opt.model_dir)
        with torch.no_grad():
            model.eval()
            val_mse = 0.
            for sample in tqdm(test_loader):
                img,label=sample
                image = sample[0].to(device)
                image = torch.nn.functional.interpolate(image, (128, 128), mode='bilinear')
                recon_combined, recons, masks, slots = model(image)
                mse_loss = criterion(recon_combined, image)

                val_mse += mse_loss.item()

            val_mse /= val_epoch_size

            visualize(image[-1],recon_combined[-1], recons[-1])
            writer.add_scalar('VAL/mse', val_mse, epoch + 1)
            print('====> Val | Epoch: {:3} \t Loss = {:F}'.format(epoch + 1, val_mse))


