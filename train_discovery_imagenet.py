import os
import argparse
from data.open_set_datasets import get_datasets,get_class_splits
import torch.optim as optim
import torch
from torchvision.utils import make_grid
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from models import model_discovery_imagenet,model_prediction
import math
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from data import Voc
from utils.evaluator import OSREvaluator
from utils import transform
import warnings

def visualize(img,slots):
    visz_list=[]
    num_slot=slots.shape[0]
    img=denormalize(img)
    visz_list.append(img)
    for i in range(num_slot):
        attention=slots[i].unsqueeze(0).repeat(3,1,1)
        visz_list.append(attention*img)
    return visz_list



def get_available_devices():
    sys_gpu = torch.cuda.device_count()

    device = torch.device('cuda:0' if sys_gpu > 0 else 'cpu')
    available_gpus = list(range(sys_gpu))
    return device, available_gpus


def train_object_discovery(args,trainloader,model):
    train_epoch_size = len(trainloader)
    #val_epoch_size=len(valloader)
    #best_val_loss=math.inf
    if args.different_lr:
        if isinstance(model, torch.nn.DataParallel):
            trainable_params = [{'params': filter(lambda p: p.requires_grad, model.module.get_decoder_params())},
                                {'params': filter(lambda p: p.requires_grad, model.module.get_slotattention_params()),
                                 'lr': args.learning_rate / 10}]
        else:
            trainable_params = [{'params': filter(lambda p: p.requires_grad, model.get_decoder_params())},
                                {'params': filter(lambda p: p.requires_grad, model.get_slotattention_params()),
                                 'lr': args.learning_rate / 10}]
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(trainable_params, lr=args.dis_lr)

    total_iter=0
    epoch_idx=0
    while total_iter<args.dis_steps:
        model.train()
        total_loss = 0
        for batch_idx, sample in enumerate(trainloader):
            if total_iter < args.warmup_steps:
                learning_rate = args.dis_lr * (total_iter /args.warmup_steps)
            else:
                learning_rate = args.dis_lr

            learning_rate = learning_rate * (args.decay_rate ** (total_iter / args.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate

            image = sample['img'].to(device)
            optimizer.zero_grad()

            slots,soft_mask,feats_recons_loss= model(image)
            loss=feats_recons_loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_iter += 1

            if batch_idx<train_epoch_size-1 and total_iter<args.dis_steps:
                del slots,soft_mask

            if total_iter > args.warmup_steps and total_iter % train_epoch_size == 0:
                writer.add_scalar('Train/mse', total_loss, total_iter)

            if total_iter>=args.dis_steps:
                break

        total_loss /= train_epoch_size
        epoch_idx+=1
        print("Train | Step: {}, Loss: {:5f} DIS_LR: {:5f} ".format(total_iter, total_loss,learning_rate))

        if epoch_idx % 5 == 0:
            img_list = []
            with torch.no_grad():
                for batch_idx, sample in enumerate(trainloader):
                    image = sample['img'].to(device)
                    slots, soft_mask, feats_recons_loss = model(image)
                    __, __, height, width = image.shape
                    soft_mask = torch.nn.functional.interpolate(soft_mask, (height, width), mode='bilinear')
                    if len(img_list) < 5:
                        img_list.append([image[-1].cpu(), soft_mask[-1].cpu()])
                    else:
                        break
                visu_list = []
                for idx in range(5):
                    img = img_list[idx][0]
                    slot_mask = img_list[idx][1]

                    visu_list.extend(visualize(img, slot_mask))
                visu_list=torch.stack(visu_list,0)
                visu_list = make_grid(visu_list, nrow=args.num_slots+1)
                writer.add_image("Discovery", visu_list, epoch_idx)
    save_path = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists:
        os.mkdir(save_path)
    save_path = os.path.join(save_path, str(args.exp_name) + "_dis.pth")
    torch.save(model.state_dict(), save_path)


def train_prediction(args,model):
    train_epoch_size = len(trainloader)
    if args.dis_exp:
        resume_path=os.path.join("checkpoints","tiny_dis.pth")
        checkpoint=torch.load(resume_path)

        msg=model.load_state_dict(checkpoint, strict=False)
        print(msg)
    if args.different_lr:
        if isinstance(model, torch.nn.DataParallel):
            trainable_params = [{'params': filter(lambda p: p.requires_grad, model.module.get_decoder_params())},
                                {'params': filter(lambda p: p.requires_grad, model.module.get_slotattention_params()),
                                 'lr': args.learning_rate / 10}]
        else:
            trainable_params = [{'params': filter(lambda p: p.requires_grad, model.get_decoder_params())},
                                {'params': filter(lambda p: p.requires_grad, model.get_slotattention_params()),
                                 'lr': args.learning_rate / 10}]
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.cls_lr)

    total_iter = 0
    epoch=0
    while True:
        model.train()
        learning_rate = args.cls_lr
        total_loss = 0
        correct, total=0,0
        evaluator.eval(model,testloader,outloader,epoch,writer)
        for batch_idx, sample in enumerate(trainloader, 0):
            total_iter += 1

            if total_iter//2000:
                learning_rate = learning_rate/2
            optimizer.param_groups[0]['lr'] = learning_rate
            image = sample['img'].to(device)
            label = sample['label'].to(device)
            class_label=sample['class_label'].to(device)
            optimizer.zero_grad()
            slots, pred, matching_loss = model(image, class_label)##mlp_pred: [b,num_slot,num_class]
            if args.use_softmax:
                pred=torch.softmax(pred,dim=-1)
            pred=torch.argmax(pred.flatten(1,2),dim=-1)
            pred%=len(args.train_classes)
            total += label.size(0)
            correct += (pred == label.data).sum()

            total_loss += matching_loss.item()

            matching_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_iter += 1
            if batch_idx < train_epoch_size - 1 and total_iter<args.cls_steps:
                del slots, pred,
            if total_iter>=args.cls_steps:
                break
        epoch+=1
        total_loss /= train_epoch_size
        acc = float(correct) * 100. / float(total)
        print("Train | Step: {}, Loss: {:5f}, Acc: {:5f}".format(total_iter, total_loss,acc))
        writer.add_scalar('train_loss', total_loss, total_iter)

    save_path=os.path.join(args.log_dir,"checkpoints")
    if not os.path.exists:
        os.mkdir(save_path)
    save_path=os.path.join(save_path,str(args.exp_name) + "_osr.pth")
    torch.save(model, save_path)


if __name__=="__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()

    parser.add_argument
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--dataset', default="voc", type=str, help='random seed')
    parser.add_argument('--exp_name', default="tiny", type=str, help='name of experiment')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--image_size', default=224, type=tuple)
    parser.add_argument('--num_slots', default=6, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hid_dim', default=256, type=int, help='hidden dimension size')
    parser.add_argument('--dis_lr', default=0.0004, type=float)
    parser.add_argument('--cls_lr', default=0.001, type=float)
    parser.add_argument('--dis_steps', default=250000, type=float)
    parser.add_argument('--cls_steps', default=15000, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=100000, type=int, help='Rate for the learning rate decay.')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
    parser.add_argument('--log_path', default='./log', type=str, help='the path of exp log')
    parser.add_argument('--different_lr', default=False, type=bool, help='use different lr to train en/decoder')
    parser.add_argument('--use_softmax', default=True, type=bool, help='whether use softmax or logits for evaluation')
    parser.add_argument('--out-num', type=int, default=10, help='For cifar-10-100')
    # aug
    parser.add_argument('--transform', type=str, default='default')
    parser.add_argument('--rand_aug_m', type=int, default=None)
    parser.add_argument('--rand_aug_n', type=int, default=None)

    parser.add_argument('--split_train_val', default=True,
                        help='Subsample training set to create validation set', metavar='BOOL')
    parser.add_argument('--use_default_parameters', default=False,
                        help='Set to True to use optimized hyper-parameters from paper', metavar='BOOL')

    parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')

    args = parser.parse_args()





    device, available_gpus=get_available_devices()
    if args.dataset in ["voc","coco","tinyimagenet"]:
        args.dis_exp=True
        dis_model=model_discovery_imagenet.Net(image_size=args.image_size,num_slot=args.num_slots,num_iteration=args.num_iterations,
                  slot_dim=args.hid_dim).to(device)
    else:
        args.dis_exp=False

    cls_model=model_prediction.Net(image_size=args.image_size,num_slot=args.num_slots,num_iteration=args.num_iterations,
              slot_dim=args.hid_dim,num_classes=6).to(device)

    if len(available_gpus)>0:
        cls_model= torch.nn.DataParallel(cls_model, device_ids=available_gpus)
        if args.dis_exp:
            dis_model = torch.nn.DataParallel(dis_model, device_ids=available_gpus)
    args.log_dir = os.path.join(args.log_path,args.exp_name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)


    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,
                                                                 cifar_plus_n=args.out_num)

    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                            open_set_classes=args.open_set_classes, balance_open_set_eval=False,
                            split_train_val=args.split_train_val, image_size=args.image_size, seed=0,
                            args=args)

    denormalize = transform.DeNormalize(std=datasets["std"],
                                                   mean=datasets["mean"])

    # ------------------------
    # RANDAUG HYPERPARAM SWEEP
    # ------------------------
    if args.transform == 'default':
        if args.rand_aug_m is not None:
            if args.rand_aug_n is not None:
                datasets['train'].transform.transforms[0].m = args.rand_aug_m
                datasets['train'].transform.transforms[0].n = args.rand_aug_n
    #
    # # ------------------------
    # # DATALOADER
    # # ------------------------
    dataloaders = {}
    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                    shuffle=shuffle, sampler=None, num_workers=0)
    #print(datasets['train'][0])
    trainloader = dataloaders['train']
    testloader = dataloaders['test_known']
    outloader = dataloaders['test_single']

    evaluator=OSREvaluator(num_known_classes=len(args.train_classes),use_softmax=args.use_softmax)
    # if args.dis_exp:
    #     dis_dataset=torch.utils.data.ConcatDataset([datasets['train'],datasets['test_known']])
    #     dis_loader=DataLoader(dis_dataset,batch_size=args.batch_size,
    #                                 shuffle=True, sampler=None, num_workers=0)
    #     train_object_discovery(args,dis_loader,dis_model)
    train_prediction(args,cls_model)








