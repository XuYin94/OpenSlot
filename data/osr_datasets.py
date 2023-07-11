"""Implementation of datasets."""
import collections
import logging
import os
from data.open_set_datasets import get_datasets
from torch.utils.data import Dataset
from distutils.util import strtobool
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class OSRDatamodule(pl.LightningDataModule):
    def __init__(self,exp_name: str="voc",
                 batch_size: int=64, num_workers: int=4, split_idx: Optional[int]=None,transform_type:Optional[str]=None):

        self.exp_name=exp_name
        self.num_workers=num_workers
        self.transform_type=transform_type
        self.batch_size=batch_size
    def setup(self):
        datasets=get_datasets(name=self.exp_name,transform=self.transform_type)
        self.train_dataset=datasets["train"]
        self.val_dataset=datasets["val"] ## val set of the in-dis samples
        self.test_dataset=datasets["test"]
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size,num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size,num_workers=self.num_workers)
