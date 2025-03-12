import os
from data.datasets import get_mixed_osr_dataset_funcs
import torch.optim as optim
import torch
from omegaconf import  OmegaConf
import hydra_zen
from torchvision.utils import make_grid
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.openslot import Net
from torch.utils.data import DataLoader
from utils.utils import get_available_devices,log_visualizations,multi_correct_slot
from torch.optim.lr_scheduler import StepLR
from datetime import datetime


def OSR_finetuning(config,model,train_loader,visualizer,device):

    exp_path=os.path.join(config.trainer.log_root_path,config.trainer.exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    log_path=os.path.join(exp_path,datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    os.mkdir(log_path)
    writer = SummaryWriter(log_path)
    params = model.get_trainable_params_groups(different_lr=False)
    hyperarameter=config.optimizers
    if len(params) > 1:
        params[0]['lr'] = 0.01
    train_epoch_size = len(train_loader)
    optimizer = optim.Adam(params, lr=hyperarameter['optimizer']["lr"])
    lr_scheduler=StepLR(optimizer,step_size=hyperarameter['lr_scheduler']["decay_epoch"]*train_epoch_size,gamma=0.5)

    #print(config.max_steps//train_epoch_size)
    current_iter = 0
    epoch=0
    if not config.Checkpoint_weight:
        weight=torch.load(config.Checkpoint_weights)
        model.load_state_dict(weight,strict=False)

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
            sample = {key: values.cuda() for key, values in sample.items()}
            outputs = model(sample)
            
            slots, logits, fg_matching_loss, bg_cls_loss = outputs["slots"], outputs["fg_pred"], outputs["fg_matching_loss"], outputs["bg_loss"]
            
            loss = fg_matching_loss + bg_cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            slot_pred = multi_correct_slot(logits).cpu().numpy()
            gt = sample['label'].to(device).cpu().numpy()
            
            gts.append(gt)
            preds.append(slot_pred)
            
            total_loss += loss.item()
            fg_loss += fg_matching_loss.item()
            bg_loss += bg_cls_loss.item()
            
            lr_scheduler.step()
            
            if batch_idx < train_epoch_size - 1 and current_iter < max_steps:
                del slots, logits
            if current_iter >= max_steps:
                break

        if epoch % trainer_config.log_every_n_epoch == 0:
            image = sample['img'].cuda()
            decoder_output = model.get_slot_attention_mask(image)
            log_visualizations(visualizer, writer, decoder_output, image, current_iter)

        epoch += 1
        total_loss /= train_epoch_size
        fg_loss /= train_epoch_size
        bg_loss /= train_epoch_size

        print(f"Train | Epoch: {epoch}, Loss: {total_loss:.5f}, Fg_loss: {fg_loss:.5f}, Bg_loss: {bg_loss:.5f}")
        writer.add_scalar('train_loss', total_loss, current_iter)

        if (epoch%trainer_config.save_every_n_epoch)==0:
            save_path=os.path.join(exp_path,trainer_config.exp_name + "_osr_epoch_"+str(epoch)+".pth")
            torch.save(model.state_dict(), save_path)
    save_path=os.path.join(exp_path,trainer_config.exp_name + "_osr_final.pth")
    torch.save(model.state_dict(), save_path)
    
    
def main_worker(config):
    device,__=get_available_devices()
    model = Net(config.models, checkpoint_path=config.Discovery_weights).to(device)
    visualization=hydra_zen.instantiate(config.visualizations, _convert_="all")

    train_set = get_mixed_osr_dataset_funcs(config.dataset.name)["train"]

    trainloader = DataLoader(train_set, batch_size=config.dataset.batch_size,
                                    shuffle=True, sampler=None, num_workers=config.dataset.num_workers)
    
    
    OSR_finetuning(config,model,trainloader,visualization,device)
if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    cfg = OmegaConf.load("configs/multi_osr_voc_config.yaml")
    import torch
    main_worker(cfg)
