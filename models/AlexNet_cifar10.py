import sys
import torch
import torch.nn as nn
import os

sys.path.insert(0, "../")



import torchvision.transforms.functional as TF
# print(os.path.abspath(os.getcwd()))
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

from utils.helper import cal_std_mean_and_get_plot_data, save_std_mean, plot_all_model, compute_mean_std

from models.base_model import Models
from models.gradient_dropconnect import GradWeightDrop as gd2
from torch.nn.modules.utils import _pair
import random
import matplotlib.pyplot as plt
from random import shuffle
from collections import OrderedDict
from torch.backends import cudnn

cudnn.benchmark = True # fast training

class AlexNet(Models):
    def __init__(self, num_classes, add_layer=None, 
                 drop_model=None, drop_connect=False, apply_lr_scr=True, 
                 normal_drop=False, p=0.5, model_name="123"):
        super().__init__()
        
        self.model_name = model_name
        self.add_layer = add_layer
        self.drop_model = drop_model
        self.drop_connect = drop_connect
        self.normal_drop = normal_drop
        self.p = p    
        
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('avgpool1', nn.AdaptiveAvgPool2d((6,6))),
        ]))

        self.linear_total = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(256*6*6, 4096)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(4096, 4096),),
            ('relu2', nn.ReLU(inplace=True)),
            ('linear3', nn.Linear(4096, num_classes))
        ]))

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.linear_total(x)
        return x
    
class AlexNet_dropout(Models):
    def __init__(self, num_classes, add_layer=None, 
                 drop_model=None, drop_connect=False, apply_lr_scr=True, 
                 normal_drop=False, p=0.5, model_name="123"):
        super().__init__()
        
        self.model_name = model_name
        self.add_layer = add_layer
        self.drop_model = drop_model
        self.drop_connect = drop_connect
        self.normal_drop = normal_drop
        self.p = p    
        
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('avgpool1', nn.AdaptiveAvgPool2d((6,6))),
        ]))

        self.linear_total = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(256*6*6, 4096)),
            ('relu1', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(p=self.p)),
            ('linear2', nn.Linear(4096, 4096),),
            ('relu2', nn.ReLU(inplace=True)),
            ('linear3', nn.Linear(4096, num_classes))
        ]))

        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.linear_total(x)
        return x
