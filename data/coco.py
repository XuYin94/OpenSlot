
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import os.path
from PIL import Image
import random
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from utils import transform,colorization

IMG_FOLDER_NAME = "train2014"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

# CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
#         'bottle', 'bus', 'car', 'cat', 'chair',
#         'cow', 'diningtable', 'dog', 'horse',
#         'motorbike', 'person', 'pottedplant',
#         'sheep', 'sofa', 'train',
#         'tvmonitor']
#
# N_CAT = len(CAT_LIST)
#
# CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

cls_labels_dict = np.load('./data/cls_labels_coco.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    s = str(int_filename).split('\n')[0]
    if len(s) != 12:
        s = '%012d' % int(s)
    return s


def load_image_label_list_from_npy(img_name_list):
    return np.array([cls_labels_dict[int(img_name)] for img_name in img_name_list])

def get_img_path(img_name, coco14_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(coco14_root, IMG_FOLDER_NAME, 'COCO_train2014_' + img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)
    img_name_list = img_name_list[::-1]

    return img_name_list



class COCO14SegmentationDataset(Dataset):

    def __init__(self, img_name_list_path, label_dir,coco14_root,transform):

        self.img_name_list = load_img_name_list(img_name_list_path)[:3000]
        self.coco14_root = coco14_root
        self.palette=colorization.COCO_palette
        self.label_dir = label_dir
        self.salience_dir=os.path.join(self.coco14_root,"sal")

        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.transform = transform
    def __len__(self):
        #print("Totally have {} samples in coco set.".format(len(self.img_name_list)))
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)
        img = Image.open(get_img_path(name_str, self.coco14_root))
        label = Image.open(os.path.join(self.label_dir, name_str.lstrip('0') + '.png'))
        #print(name_str)
        salience = Image.open(os.path.join(self.salience_dir, 'COCO_train2014_'+name_str+ '.png')).convert("L")
        if self.transform is not None:
            sample=self.transform({'img':img,'label':label,'salience':salience})
        sample['name']=name_str
        sample['label_cls']=torch.from_numpy(self.label_list[idx])
        return sample


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    args=parser.parse_args()
    trans=transform.Compose([transform.HorizontalFilp(),
                             transform.Crop(base_size=256,crop_height=224,crop_width=224,type='central'),
                             transform.GaussianBlur(),
                             transform.ToTensor(),
                             transform.Normalize(std=[0.229, 0.224, 0.225],
                                               mean=[0.485, 0.456, 0.406])
                             ])

    split='train_aug'
    data_root="D:\\datasets\\coco\\2014\\"
    data_list="D:\\datasets\\coco\\2014\\train14.txt"
    gt_mask="D:\\datasets\\coco\\2014\\mask\\"
    COCO_set=COCO14SegmentationDataset(img_name_list_path=data_list, label_dir=gt_mask,coco14_root=data_root,transform=trans)
    col_map=COCO_set.palette
    dataloader = DataLoader(COCO_set, batch_size=10, shuffle=True, num_workers=0)

    val_list=[]
    for ii, sample in enumerate(dataloader):
        gt = sample['label'][0]
        if (len(torch.unique(gt))>3):
            if len(val_list)<3:
                gt=colorization.colorization(gt.numpy(),col_map)
                img=sample['img'][0]
                #print(img.max())
                img=transform.DeNormalize(std=[0.229, 0.224, 0.225],
                                               mean=[0.485, 0.456, 0.406])(img)
                gt=transforms.ToTensor()(gt.convert('RGB'))
                val_list.extend([img,gt])
            else:
                break
    val_list=torch.stack(val_list,0)
    val_list=make_grid(val_list,nrow=2,padding=5)
    val_list=transforms.ToPILImage()(val_list)
    val_list.show()