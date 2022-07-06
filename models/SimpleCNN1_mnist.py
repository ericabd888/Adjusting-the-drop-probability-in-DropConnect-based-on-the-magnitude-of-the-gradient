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


class SimpleCNN1(Models):
    def __init__(self, num_classes, add_layer=None, 
                 drop_model=None, drop_connect=False, apply_lr_scr=True, 
                 normal_drop=False, p=0.5, model_name="SimpleCNN1"):
        super().__init__()
        self.model_name = model_name
        self.add_layer = add_layer
        self.drop_model = drop_model
        self.drop_connect = drop_connect
        self.normal_drop = normal_drop
        self.p = p      
        
        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1)),
            ('maxpool1', nn.MaxPool2d(3,2)),
            ('conv2', nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1)),
            ('relu1', nn.ReLU()),
            ('avgpool1', nn.AvgPool2d(3,2)),
            ('conv3', nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1)),
            ('relu1', nn.ReLU()),
            ('avgpool2', nn.AvgPool2d(3,2)),
        ]))
        self.linear_total = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(256, 64)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(64, 10),)
        ]))

        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_total(x)
        return x
    
class SimpleCNN1_dropout(Models):
    def __init__(self, num_classes, add_layer=None, 
                 drop_model=None, drop_connect=False, apply_lr_scr=True, 
                 normal_drop=False, p=0.5, model_name="SimpleCNN1"):
        super().__init__()
        self.model_name = model_name
        self.add_layer = add_layer
        self.drop_model = drop_model
        self.drop_connect = drop_connect
        self.normal_drop = normal_drop
        self.p = p      
        
        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1)),
            ('maxpool1', nn.MaxPool2d(3,2)),
            ('conv2', nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1)),
            ('relu1', nn.ReLU()),
            ('avgpool1', nn.AvgPool2d(3,2)),
            ('conv3', nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1)),
            ('relu1', nn.ReLU()),
            ('avgpool2', nn.AvgPool2d(3,2)),
        ]))
        self.linear_total = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(256, 64)),
            ('Dropout',nn.Dropout(p=self.p)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(64, 10),)
        ]))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_total(x)
        return x

LR = 0.001
weight_decay = 5e-4
mini_lr = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)


small_weight_drop = gd2(DEVICE, I_P=0.4, W_P=0.25, w_small=True, name="First")
big_weight_drop = gd2(DEVICE, I_P=0.4, W_P=0.25, w_small=False, name="First")
small_grad_drop = gd2(DEVICE, I_P=0.4, GD_P=0.25, gd_small=True, name="First")
big_grad_drop = gd2(DEVICE, I_P=0.4, GD_P=0.25, gd_small=False, name="First")

setting_dict_list = [{'vgg_name':'simple_cnn1', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
                      'model_name':'Vallina'},
                    {'vgg_name':'simple_cnn1', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
                     'model_name':'DropOut'},
                    {'vgg_name':'simple_cnn1', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
                    'drop_connect':True, 'normal_drop':True, 'model_name':'DropConnect', 'p':0.5},
                    {'vgg_name':'simple_cnn1', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
                    'drop_model':[small_weight_drop], 'drop_connect':True, 'model_name':'DropSmallW'},
                    {'vgg_name':'simple_cnn1', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
                    'drop_model':[big_weight_drop], 'drop_connect':True, 'model_name':'DropBigW'},
                    {'vgg_name':'simple_cnn1', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
                    'drop_model':[small_grad_drop], 'drop_connect':True, 'model_name':'DropSmallGD'},
                    {'vgg_name':'simple_cnn1', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
                    'drop_model':[big_grad_drop], 'drop_connect':True, 'model_name':'DropBigGD'},]




# model = SimpleCnn1(**setting_dict_list[0])
# model.to(DEVICE)
# model.weights_init()
# model.compiler(lr=LR, weight_decay=weight_decay, lr_scheduler_apply=True, 
#                            device=DEVICE, cosine=True, max_lr=LR, first_cycle_steps=120, warmup_steps=0, gamma=0.2, warm_up_lr_apply=True, warmup_iter=12)
# model.fit(120, train_loader, valid_loader, test_loader, show_test_result=True)
# print(model._Eval_Score(save=False))

# Exp_time = 1
# RUN_EPOCH = 100

# compute_all = []


# FILE = "./Model_pool/05_04"
# # MODEL = "/cifar100_vgg16_vallina"
# # pic_FILE = FILE + MODEL + ".png"
# # model_FILE = FILE + MODEL + ".pt"
# model_name = []
# for m in tzip(range(len(setting_dict_list))):
#     temp_list = []
#     for time in tqdm(range(Exp_time)):
#         if m[0] == 1:
#             model = SimpleCnn1_dropout(**setting_dict_list[m[0]])
#             model.to(DEVICE)
#             model.weights_init()
#             model.compiler(lr=LR, weight_decay=weight_decay, lr_scheduler_apply=True, 
#                                        device=DEVICE, cosine=True, max_lr=LR, first_cycle_steps=100, 
#                            warmup_steps=0, gamma=0.2, warm_up_lr_apply=True, warmup_iter=12)
#             model.fit(RUN_EPOCH, train_loader, valid_loader, test_loader, show_test_result=True)
#         else:
#             model = SimpleCnn1(**setting_dict_list[m[0]])
#             model.to(DEVICE)
#             model.weights_init()
#             model.compiler(lr=LR, weight_decay=weight_decay, lr_scheduler_apply=True, 
#                                        device=DEVICE, cosine=True, max_lr=LR, first_cycle_steps=100, 
#                            warmup_steps=0, gamma=0.2, warm_up_lr_apply=True, warmup_iter=12)
#             model.fit(RUN_EPOCH, train_loader, valid_loader, test_loader, show_test_result=True)
#             if model.drop_model:
#                 for drop in model.drop_model:
#                     drop._reset_time()
#         if time == 0:
#             model_name.append(model.model_name)
#             print(model._Eval_Score(picture_location=FILE+'/mnist_simplecnn1_exp_'+model.model_name+'.pdf'))
#             model._save(FILE+'/mnist_simplecnn1_exp_'+model.model_name+'.pt', total_transform)
#         temp_list.append(model.model_info)
#     compute_all.append(temp_list)