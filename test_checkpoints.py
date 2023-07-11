import torch
import hydra
from typing import Any, Dict, Optional
import hydra_zen
import dataclasses
import yaml
import  torch.nn as nn
from omegaconf import DictConfig, OmegaConf

cfg=OmegaConf.load("configs/config.yaml")
from ocl.utils.routing import Combined
models = hydra_zen.instantiate(cfg.models, _convert_="all")
models=nn.ModuleDict(models)
print(models)
# for name, module in models.items():
#     print(name)
# if isinstance(models, Dict):
#     models = Combined(**models)
# for name, module in models.items():
#     print(name)
checkpoint=torch.load("checkpoints/model_final.ckpt")["state_dict"]
for key in list(checkpoint.keys()):
    if 'models.' in key:
        checkpoint[key.replace('models.', '')] = checkpoint[key]
        del checkpoint[key]
models.load_state_dict(checkpoint,strict=True)
