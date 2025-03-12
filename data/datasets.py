import os
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from data.augmentations.cut_out import *
from data.augmentations.randaugment import RandAugment
import torchvision

# Root paths for datasets
root_paths = {
    "coco": "/mnt/nas4/yinxu/Dataset/coco",
    "voc": "/mnt/nas4/yinxu/Dataset/VOCdevkit/VOC2012",
    "voc_test": "/mnt/nas4/yinxu/Dataset/VOCdevkit/test/VOCdevkit/VOC2012",
}

# Dataset configurations
dataset_configs = {
    'voc-6-14': {
        'root_path': root_paths['voc'],
        'split_path': "mixed/single/voc-6-14",
        'nbr_known': 6,
        'nbr_unknown': 14,
        'nbr_slot': 6,
        'max_num_object': 1
    },
    'coco-20-60': {
        'root_path': root_paths['coco'],
        'split_path': "mixed/single/coco-20-60",
        'nbr_known': 20,
        'nbr_unknown': 60,
        'nbr_slot': 7,
        'max_num_object': 1
    },
    "voc-coco": {
        'root_path': root_paths['voc'],
        'split_path': "mixed/multi/voc-coco",
        'nbr_known': 20,
        'nbr_unknown': 60,
        'nbr_slot': 6,
        'max_num_object': 6
    },
    "intra-coco-20-60": {
        'root_path': root_paths['coco'],
        'split_path': "mixed/multi/intra-coco/Task1",
        'nbr_known': 20,
        'nbr_unknown': 60,
        'nbr_slot': 7,
        'max_num_object': 7
    },
    "intra-coco-40-40": {
        'root_path': root_paths['coco'],
        'split_path': "mixed/multi/intra-coco/Task2",
        'nbr_known': 40,
        'nbr_unknown': 40,
        'nbr_slot': 7,
        'max_num_object': 7
    },
    "intra-coco-60-20": {
        'root_path': root_paths['coco'],
        'split_path': "mixed/multi/intra-coco/Task3",
        'nbr_known': 60,
        'nbr_unknown': 20,
        'nbr_slot': 7,
        'max_num_object': 7
    }
}

def get_mixed_osr_dataset_funcs(exp_name):
    """
    Returns dataset functions for mixed OSR experiments.
    """
    train_transform, test_transform = get_transform(exp_name=exp_name)
    configs = dataset_configs[exp_name]
    return Get_OSR_Datasets(configs, train_transform, test_transform, exp=exp_name)

def Get_OSR_Datasets(configs, train_transform, test_transform, exp="voc"):
    """
    Returns datasets for training, validation, and testing.
    """
    DatasetClass = Multi_label_OSR_dataset if exp == "voc-coco" or "intra" in exp else OSRDataset

    train_set = DatasetClass(configs, prefix='train', exp_name=exp, transform=train_transform)
    val_set = DatasetClass(configs, prefix='val', exp_name=exp, transform=test_transform)
    no_mixture_test_set = DatasetClass(configs, prefix='no_mixture_test', exp_name=exp, transform=test_transform)
    mixture_test_set = DatasetClass(configs, prefix='all_mixture_test', exp_name=exp, transform=test_transform)

    print('Train: ', len(train_set), 'Positives: ', len(val_set), 'no_mixture_negatives: ', len(no_mixture_test_set), 'mixed_negatives: ', len(mixture_test_set))

    return {
        'train': train_set,
        'postives': val_set,
        'no_mixed_neg': no_mixture_test_set,
        'mixed_neg': mixture_test_set
    }

def get_transform(exp_name, image_size=224, args=None):
    """
    Returns transformations for training and testing.
    """
    if "voc" in exp_name or "coco" in exp_name:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.5, 2.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(1, 9, args=args),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return train_transform, test_transform

class OSRDataset(Dataset):
    def __init__(self, configs, prefix="train", transform=None, exp_name="voc-6-14"):
        """
        Open Set Recognition (OSR) Dataset.
        """
        self.root_dir = configs['root_path']
        self.split_path = configs['split_path']
        self.prefix = prefix
        self.exp = exp_name
        self.transform = transform
        self.name_list, self.labels = self._get_name_list()
        self.max_num_object = configs['max_num_object']
        self.num_known_classes = configs['nbr_known']
        self.num_slot = configs['nbr_slot']

    def _get_name_list(self):
        file_path = os.path.join("./data/osr_splits", self.split_path, f"{self.prefix}.txt")
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise ValueError(f"File {file_path} not found.")

        print(f"Totally have {len(lines)} samples in {self.prefix} set.")
        img_paths, labels = [], []

        for line in lines:
            name, label = line.split()
            img_paths.append(name)
            labels.append(int(label))

        return img_paths, labels

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        img_path = os.path.join(self.root_dir, name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise ValueError(f"Image {img_path} not found.")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.long)
        sample = {'img': image, 'label': label}

        if self.prefix == "train":
            class_label = torch.zeros((self.max_num_object,), dtype=torch.int64)
            selected_slots = torch.zeros((self.max_num_object,), dtype=torch.float32)
            class_label[0] = label
            selected_slots[0] = 1
            #print(class_label.shape)
            class_label = torch.nn.functional.one_hot(class_label, num_classes=self.num_known_classes)
            #print(class_label.shape)
            sample.update({
                'class_label': class_label,
                'fg_channel': selected_slots.unsqueeze(1)
            })
        # print(sample['fg_channel'].shape)
        return sample

class Multi_label_OSR_dataset(Dataset):
    def __init__(self, configs, prefix='train', exp_name="voc", transform=None):
        self.root_dir = configs["root_path"]
        self.transform = transform
        self.prefix = prefix
        self.exp = exp_name
        self.exp_info = os.path.join('./data/osr_splits', configs['split_path'], prefix)

        self.img_list = open(f'{self.exp_info}.txt').readlines()
        if prefix=="train":
            self.cls_labels_dict = np.load(f'{self.exp_info}_label.npy', allow_pickle=True)

        self.max_num_object = configs['max_num_object']
        self.num_known_classes = configs['nbr_known']
        self.num_slot = configs['nbr_slot']

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        name = self.img_list[index].strip()
        img_path = os.path.join(self.root_dir, name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise ValueError(f"Image {img_path} not found.")

        if self.transform:
            image = self.transform(image)

        sample = {'img': image}

        if self.prefix in ["train"]:
            sample['label'] = self.cls_labels_dict[index]
            label = self.cls_labels_dict[index].nonzero()[0][:self.max_num_object]
            nbr_clss = min(len(label), self.max_num_object)

            selected_slots = torch.zeros((self.max_num_object))
            semantic_label = torch.zeros((self.max_num_object))
            selected_slots[:nbr_clss] = 1
            semantic_label[:nbr_clss] = torch.as_tensor(np.array(label), dtype=torch.long) + 1

            sample['fg_channel'] = selected_slots.unsqueeze(1)
            semantic_label = torch.nn.functional.one_hot(torch.as_tensor(semantic_label, dtype=torch.int64), num_classes=self.num_known_classes + 1)[:, 1:]
            sample['class_label'] = semantic_label
        else:
            sample['label'] = torch.tensor(float('inf'))
        # print(sample['class_label'].shape)
        # print(sample['fg_channel'].shape)
        return sample
