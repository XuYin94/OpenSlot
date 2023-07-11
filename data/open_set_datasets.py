from data.cifar import get_cifar_10_10_datasets, get_cifar_10_100_datasets
from data.OSR_tiny_imagenet import get_tiny_image_net_datasets
from data.svhn import get_svhn_datasets
from data.mnist import get_mnist_datasets
from data.osr_splits.osr_splits import osr_splits
from data.augmentations import get_transform
from data.osr_natural import  Get_OSR_Datasets
from config import osr_imagenet_split_dir
from torch.utils.data import DataLoader
import argparse
import os
import sys
import pickle
import torch

"""
For each dataset, define function which returns:
    training set
    validation set
    open_set_known_images
    open_set_unknown_images
"""

get_dataset_funcs = {
    'cifar-10-100': get_cifar_10_100_datasets,
    'cifar-10-10': get_cifar_10_10_datasets,
    'mnist': get_mnist_datasets,
    'svhn': get_svhn_datasets,
    'tinyimagenet': get_tiny_image_net_datasets
}

def get_datasets(name, transform='default', image_size=224, train_classes=(0, 1, 8, 9),
                 open_set_classes=range(10), balance_open_set_eval=False, split_train_val=True, seed=0, args=None):

    """
    :param name: Dataset name
    :param transform: Either tuple of train/test transforms or string of transform type
    :return:
    """

    print('Loading datasets...')
    if isinstance(transform, tuple):
        train_transform, test_transform = transform
    else:
        train_transform, test_transform = get_transform(name,transform_type=transform, image_size=image_size, args=args)

    if name in get_dataset_funcs.keys():
        datasets = get_dataset_funcs[name](train_transform, test_transform,
                                  train_classes=train_classes,
                                  open_set_classes=open_set_classes,
                                  balance_open_set_eval=balance_open_set_eval,
                                  split_train_val=split_train_val,
                                  seed=seed)
        datasets["mean"]= [0.5, 0.5, 0.5]
        datasets["std"] = [0.5, 0.5, 0.5]
    elif name in ["coco", "voc", "nus"]:
        root_paths = {
            "coco": "D:\\datasets\\coco\\2014\\",
            "voc": "D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\",
            "nus": "D:\\datasets\\nus_wide\\"
        }
        root_dir = root_paths[name]
        datasets = Get_OSR_Datasets(train_transform, test_transform, dataroot=root_dir, exp=name)

        datasets["mean"]= [0.485, 0.456, 0.406]
        datasets["std"] = [0.229, 0.224, 0.225]
    else:
        raise NotImplementedError


    return datasets

def get_class_splits(dataset, split_idx=0, cifar_plus_n=10):

    if dataset in ('cifar-10-10', 'mnist', 'svhn'):
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(10) if x not in train_classes]

    elif dataset == 'cifar-10-100':
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = osr_splits['cifar-10-100-{}'.format(cifar_plus_n)][split_idx]

    elif dataset == 'tinyimagenet':
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(200) if x not in train_classes]
    elif dataset == 'voc':
        train_classes = [0, 2, 6, 7, 11, 14]
        open_set_classes = [1, 3, 4, 5, 8, 9, 12, 13, 15, 16, 17, 18, 19]
    elif dataset == 'coco':
        train_classes = [4, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 61, 74, 75, 77]
        open_set_classes = [0, 1, 2, 3, 5, 7, 12, 16, 25, 28, 33, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 62, 65,
                            67, 68, 69, 71, 72, 76, 79]
    elif dataset == 'nus':
        train_classes = [0, 1, 2, 4, 7, 9, 13, 14, 16, 17, 21, 22, 40, 50, 59]
        open_set_classes = [3, 5, 6, 8, 10, 12, 15, 18, 19, 20, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37,
                            38, 39,
                            41, 42, 43, 44, 46, 47, 48, 51, 53, 54, 55, 57, 58, 62, 64, 66, 71, 72, 74, 78]
    else:
        raise  NotImplementedError
    return train_classes, open_set_classes

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__=="__main__":

    parser = argparse.ArgumentParser("Training")

    # Dataset
    parser.add_argument('--dataset', default='tinyimagenet',type=str, help="")
    parser.add_argument('--out-num', type=int, default=10, help='For cifar-10-100')
    parser.add_argument('--image_size', type=int, default=224)

    # optimization
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num-centers', type=int, default=1)


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

    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,
                                                                 cifar_plus_n=args.out_num)
    print(args.train_classes)
    print(args.open_set_classes)
    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                            open_set_classes=args.open_set_classes, balance_open_set_eval=False,
                            split_train_val=args.split_train_val, image_size=args.image_size, seed=0,
                            args=args)


    # ------------------------
    # RANDAUG HYPERPARAM SWEEP
    # ------------------------
    if args.transform == 'rand-augment':
        if args.rand_aug_m is not None:
            if args.rand_aug_n is not None:
                datasets['train'].transform.transforms[0].m = args.rand_aug_m
                datasets['train'].transform.transforms[0].n = args.rand_aug_n

    # ------------------------
    # DATALOADER
    # ------------------------
    #print(datasets['train'].targets)
    #print(datasets['train'].samples)
    dataloaders = {}
    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                    shuffle=shuffle, sampler=None, num_workers=0)

    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']
    from tqdm import tqdm
    for sample in tqdm(trainloader):

        print(sample["label"])