import os
import xml.etree.ElementTree as ET
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageFont
import torch
import torchvision
import numpy as np
import cv2
from torchvision.utils import draw_bounding_boxes
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import hydra
from omegaconf import DictConfig, OmegaConf
import hydra_zen
from torch.utils.data import Dataset,DataLoader
from utils.evaluator import OSREvaluator
from ocl.visualizations import masks_to_boxes
from utils.utils import get_available_devices,multi_correct_slot,log_visualizations
from models.openslot import Net
from datetime import datetime
from utils.utils import slot_energy,slot_max
from os import listdir
from torch.utils.tensorboard import SummaryWriter
from os.path import isfile, join

voc_labels_name=['aeroplane', "Bicycle", 'bird', "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
                           'cow', "Diningtable", "Dog", "Horse", "Motorbike", 'person', "Pottedplant", 'sheep', "Sofa",
                           "Train", "TVmonitor"]
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

unknown_class_color=()
known_class_color=()

def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def draw_bbox(images,predictions,names,dataset_name):
    batch_size=images.shape[0]
    for i in range(batch_size):
        if predictions[i] is not None:
            bbox_list=predictions[i]['bbox']
            label_list=predictions[i]['class']
            color_list=predictions[i]['color_list']
            rendered_image=to_pil_image(draw_bounding_boxes(images[i],bbox_list,label_list,width=3,colors=color_list,font="./arial.ttf",font_size=16))
            rendered_image.save("/mnt/nas4/yinxu/Dataset/detection_result/"+dataset_name+"\\"+str(names[i]))




def Get_detection_results(logits,bg_logits,valid_slots,attention_masks):
    detection_result=[]
    # bg_minimum = torch.min(bg_logits, dim=1, keepdim=True)[0]
    # bg_maxmim = torch.max(bg_logits, dim=1, keepdim=True)[0]
    # bg_pred = (bg_logits - bg_minimum) / (bg_maxmim - bg_minimum + 1e-10)  ## min-max normalization
    # #print(bg_pred.min())
    # valid_fg_slots = ((bg_pred < 0.001)[:,:,0] & (valid_slots>0)).bool()
    #print(valid_fg_slots.shape)
    for i in range(logits.shape[0]):
        # if valid_fg_slots[i].sum() > 0:
        #     fg_slot = logits[i, valid_fg_slots[i]]
        #     detected_object=attention_masks[i,valid_fg_slots[i]]
        # else:
        #     fg_slot, indices = torch.max(logits[i], dim=-1)
        #     detected_object=attention_masks[i,indices]
        detected_object=attention_masks[i]
        detection={}
        detection['class'] = []
        detection['bbox'] = []
        masks=(detected_object>0.75).bool()
        fg_slot=[]
        mask_list=[]
        ood_score=[]
        for idx, single in enumerate(masks):
            if (single.sum()>500):
                fg_slot.append(logits[i,idx])
                ood_score.append(torch.logsumexp(logits[i,idx], dim=-1))
                mask_list.append(single)

        if len(fg_slot)>0:
            fg_slot=torch.stack(fg_slot,dim=0)
            mask_list=torch.stack(mask_list,dim=0)
            ood_score=torch.stack(ood_score,dim=0)
            class_labels=[]
            color_list=[]
            for idx,slot in enumerate(fg_slot):
                if ood_score[idx]<=5:
                    class_labels.append("unknown")
                    color_list.append((255,0,0))
                else:
                    class_indices=torch.argmax(slot,dim=-1)
                    class_labels.append(voc_labels_name[class_indices])
                    color_list.append((255,140,0))
            # class_indices=torch.argmax(fg_slot, dim=-1)
            # class_labels=[voc_labels_name[j] for j in class_indices ]
            results= masks_to_boxes(mask_list)

            detection['class']=class_labels
            detection['bbox']=results
            detection['color_list']=color_list
            assert results.shape[0]==len(class_labels)
            detection_result.append(detection)
        else:
            detection_result.append(None)
        # for j in range(detected_object.shape[0]):
        #
        #     mask=np.uint8((detected_object[j]>0.5).cpu().numpy())
        #     masks=(detected_object[j]>0.5).bool().unsqueeze(0)
        #
        #     if masks.sum()>60:
        #         print(masks.sum())
        #         print(masks.shape)
        #
        #         print(results)
        #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #         if len(contours) > 0:
        #
        #             x, y, w, h = cv2.boundingRect(contours[0])
        #             detection['bbox'].append(torch.as_tensor([x, y, x + w, y + h]))
        # if len(detection['bbox'])>0:

        #     #print(detection['bbox'].shape)
        #     assert len(detection['class'])==len(detection['bbox'])

    return detection_result


class Out_test_Dataset(Dataset):

    def __init__(self,root_path="/mnt/nas4/yinxu/Dataset",split="coco",transform=None):
        self.root_path=root_path
        self.split=split
        self.image_list=self._load_img_names()
        self.transform=transform

    def _load_img_names(self):
        mypath=os.path.join(self.root_path,"id_voc_ood_"+str(self.split))
        return [f for f in listdir(mypath) if isfile(join(mypath, f))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_path, "id_voc_ood_"+str(self.split), self.image_list[idx])
        image = Image.open(img_name).convert('RGB')
        #print(self.image_list[idx])
        if self.transform is not None:
            image=self.transform(image)

        return {"img":image,"name":self.image_list[idx]}

class PascalVOCDetectionDataset(Dataset):
    def __init__(self, root_dir="/mnt/nas4/yinxu/Dataset/VOCdevkit/", year=["2007"],split="trainval", transform=None):
        self.root_dir = root_dir
        self.split=split
        self.year=year
        self.transform = transform
        self.common_init()
        self.image_list,self.labels = self._load_names_labels()


    def _load_names_labels(self):
        name_list=[]
        labels=[]
        image_labels=[]
        with open("./val_set.text","w") as file:

            for y in self.year:
                set="VOC"+y
                with open(os.path.join(self.root_dir, str(set), 'ImageSets', 'Main', str(self.split)+'.txt')) as f:
                    lines = f.readlines()
                prefix=self.root_dir+ str(set)+'/JPEGImages/'

                for line in lines:
                    single_label = np.zeros(20)
                    line=line.strip()
                    name_list.append(prefix+line+".jpg \n")
                    str_line=prefix+line+".jpg"
                    file.writelines('./'+str(str_line[26:])+"\n")
                    #print(prefix+line+".jpg \n")
                    annotation_path=os.path.join(self.root_dir,str(set),'Annotations',''+str(line)+'.xml')
                    result=self.parse_annotation(annotation_path)
                    labels.append(result)
                    for object in result:
                        #print(single_label)
                        single_label[object["label"]-1]=1

                    image_labels.append(single_label)
        #image_labels=np.stack(image_labels,axis=0)
        #np.save("./val_label.npy",image_labels)
        return name_list,labels


    def common_init(self):
        # init that must be shared among all subclasses of this method
        self.label_type = ['none', 'aeroplane', "Bicycle", 'bird', "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
                           'cow', "Diningtable", "Dog", "Horse", "Motorbike", 'person', "Pottedplant", 'sheep', "Sofa",
                           "Train", "TVmonitor"]
        self.convert_id = ['background', 'Aeroplane', "Bicycle", 'Bird', "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
                           'Cow', "Dining table", "Dog", "Horse", "Motorbike", 'Person', "Potted plant", 'Sheep',
                           "Sofa", "Train", "TV/monitor"]
        self.convert_labels = {}
        for idx, x in enumerate(self.label_type):
            self.convert_labels[x.lower()] = idx

        self.num_classes = len(self.label_type)  # 20 + 1(none)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'JPEGImages', self.image_list[idx].strip())
        image = Image.open(img_name).convert('RGB')
        targets = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        #print(np.asarray(image).shape)
        return image,targets

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        targets = []
        for obj in root.findall('object'):
            target = {}
            target['label'] = self.convert_labels[obj.find('name').text]
            bndbox = obj.find('bndbox')
            target['bbox'] = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            ]
            targets.append(target)
        return targets
    def collate_fn(self, batch):

        images = list()
        targets=list()
        for b in batch:
            images.append(b[0])
            targets.append(b[1])

        images = torch.stack(images, dim=0)

        return {"img":images,"label":targets}  # tensor

def inference(config):
    device, available_gpus = get_available_devices()
    model = Net(config.models, checkpoint_path=config.Discovery_weights).to(device)
    path= "tmp_checkpoints/voc_detection_osr_final_1115.pth"
    weight=torch.load(path)
    model.load_state_dict(weight,strict=False)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        # transforms.CenterCrop((224, 224)),
        transforms.Normalize(std=[0.229, 0.224, 0.225],
                             mean=[0.485, 0.456, 0.406])
    ])
    visualization=hydra_zen.instantiate(config.visualizations, _convert_="all")
    val_dataset=PascalVOCDetectionDataset(year=['2007'],split='test',transform=img_transform)
    in_dataloader=DataLoader(val_dataset,num_workers=0,batch_size=8,shuffle=False,collate_fn=val_dataset.collate_fn)
    out_coco_dataset=Out_test_Dataset(split="coco",transform=img_transform)
    out_openImage_dataset=Out_test_Dataset(split="openimages",transform=img_transform)
    out_loader_list={}
    out_loader_list["coco"]=DataLoader(out_coco_dataset,num_workers=0,batch_size=64,shuffle=False)
    out_loader_list["openimages"]=DataLoader(out_openImage_dataset,num_workers=0,batch_size=64,shuffle=False)
    exp_path="./log/object_detection/"
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    log_path=os.path.join(exp_path,datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    #print(log_path)
    os.mkdir(log_path)
    writer = SummaryWriter(log_path)


    close_pred={}
    model.eval()
    evaluator=OSREvaluator(train_loader=in_dataloader,visualizer=visualization,num_known_classes=20,exp_type="single")
    #overall_test(evaluator, model, in_dataloader, out_loader_list, 0, writer)

    import time
    start_time=time.time()
    for key, out_loader in out_loader_list.items():

        for idx, sample in enumerate(out_loader):

            images=sample["img"].cuda()
            names=sample["name"]
            outputs = model({'img':images})

            slots,logits,bg_logits,valid_slots = \
                outputs["slots"],outputs["fg_pred"],outputs["bg_pred"],outputs["valid_slots"]

            ## obtain the slot-based object proposal
            decoder_output = model.get_slot_attention_mask(images,sorted=False)
            log_visualizations(visualization,writer,decoder_output,images,idx)
            #print(decoder_output.masks.shape)
            masks=decoder_output.masks.reshape(slots.shape[0],7,14,14)
            masks=F.interpolate(masks,(224,224),mode='bilinear')
            predicted_results=Get_detection_results(logits,bg_logits,valid_slots,masks)
            images=denormalize(images)*255
            images=images.to(torch.uint8)
            draw_bbox(images,predicted_results,names,key)
    print(time.time()-start_time)



def overall_test(evaluator,model, val_loader, test_loader_list, epoch, writer):
    print("fuck")
    evaluator.eval(model, val_loader, test_loader_list, epoch, writer, processor=["slotmax","slotenergy"],compute_acc=False,oscr=False)

if __name__=="__main__":
    cfg = OmegaConf.load("configs/multi_label_voc_detection.yaml")
    inference(cfg)
    #train_dataset=PascalVOCDetectionDataset(year=['2007','2012'])
    #val_dataset=PascalVOCDetectionDataset(year=['2007'],split='test')
    #out_coco_dataset=Out_test_Dataset(split="coco")
    #out_openImage_dataset=Out_test_Dataset(split="openimages")
    # print(len(train_dataset))
    # print(len(val_dataset))
    #print(len(out_coco_dataset))
    #print(len(out_openImage_dataset))