import sys
import torch
from os.path import abspath, dirname
import os

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import pickle
import models.AlexNet_norb as AlexNet

import argparse
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

from utils.helper import cal_std_mean_and_get_plot_data, save_std_mean, plot_all_model, get_result_table

from models.base_model import Models
from models.gradient_dropconnect import GradWeightDrop as gd2
from torch.nn.modules.utils import _pair
import random
import matplotlib.pyplot as plt
from random import shuffle
from collections import OrderedDict
from torch.backends import cudnn
cudnn.benchmark = True # fast training

# Hyperparameter setting
def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('--epoch', default=75, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--dropconnect_rate', default=0.5, type=float)
    parser.add_argument('--gd_init_droprate', default=0.3, type=float)
    parser.add_argument('--gd_droprate', default=0.5, type=float)
    parser.add_argument('--w_init_droprate', default=0.3, type=float)
    parser.add_argument('--w_droprate', default=0.5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--apply_lr_scheduler', dest='flag', action='store_true')
    parser.add_argument('--no-apply_lr_scheduler', dest='flag', action='store_false')
    parser.add_argument('--warmup_steps', default=8, type=int)
    parser.set_defaults(flag=True)
    return parser
    
parser = get_parser()
args = parser.parse_args()

RUN_EPOCHS = args.epoch
LR = args.lr
weight_decay = args.weight_decay
dropout_rate = args.dropout_rate
dropconnect_rate = args.dropconnect_rate
GD_init_droprate = args.gd_init_droprate
GD_droprate = args.gd_droprate
W_init_droprate = args.w_init_droprate
W_droprate = args.w_droprate
BATCH_SIZE = args.batch_size
apply_lr_scheduler = args.flag
if apply_lr_scheduler:
    warmup_steps = args.warmup_steps
else:
    warmup_steps = 0


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        y = y.type(torch.LongTensor)
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    
class MyDataset_from_numpy(Dataset):
    def __init__(self, subset):
        self.subset = subset
        
    def __getitem__(self, index):
        x, y = self.subset[index]
#         x = torch.from_numpy(x)
        y = torch.tensor(y)
        return x, y
    def __len__(self):
        return len(self.subset)
    

file_loc = dirname(dirname(dirname(abspath(__file__))))

with open(file_loc+"/data/big_norb_data/final_train_data.pk", 'rb') as f:
    train_set = pickle.load(f)
    
with open(file_loc+"/data/big_norb_data/final_test_data.pk", 'rb') as f:
    test_set = pickle.load(f)

train_set = MyDataset_from_numpy(train_set)

test_set = MyDataset_from_numpy(test_set)    
    
    


total_data_len = len(train_set)
target = [i for i in range(total_data_len)]
shuffle(target)
train_idx = target[:int(total_data_len*0.9)]
valid_idx = target[int(total_data_len*0.9):]


training_set = torch.utils.data.Subset(train_set, train_idx)
validating_set = torch.utils.data.Subset(train_set, valid_idx)

mean1, std1 = (0.379, 0.385), (0.265, 0.269)


total_transform = {
    'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(133),
                transforms.Normalize(mean1, std1)
            ]),
    'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(133),
                transforms.Normalize(mean1, std1),
            ]),
    'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(133),
                transforms.Normalize(mean1, std1),
            ]),
} 


training_set = MyDataset(training_set, transform=total_transform['train'])
validating_set = MyDataset(validating_set, transform=total_transform['valid'])
testing_set = MyDataset(test_set, transform=total_transform['test'])

train_loader = DataLoader(training_set, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=5)

valid_loader = DataLoader(validating_set, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=5)


test_loader = DataLoader(testing_set, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=5)



drop_small_w = gd2(DEVICE, I_P=W_init_droprate, W_P=W_droprate, w_small=True, name="DropSmallW")
drop_big_w = gd2(DEVICE, I_P=W_init_droprate, W_P=W_droprate, w_small=False, name="DropBigW")
drop_small_gd = gd2(DEVICE, I_P=GD_init_droprate, GD_P=GD_droprate, gd_small=True, name="DropSmallGd")
drop_big_gd = gd2(DEVICE, I_P=GD_init_droprate, GD_P=GD_droprate, gd_small=False, name="DropBigGd")

hyper_param_configs = [{'model_name': 'Vallina', 'num_classes': 6, 'add_layer': ['linear_total.linear1']}, 
                       {'model_name': 'Dropout', 'num_classes': 6, 'add_layer': ['linear_total.linear1'],
                        'p': dropout_rate}, 
                       {'model_name': 'DropConnect', 'num_classes': 6, 'add_layer': ['linear_total.linear1'], 
                        'drop_connect': True, 'normal_drop': True, 'p': dropconnect_rate}, 
                       {'model_name': 'DropSmallW', 'num_classes': 6, 'add_layer': ['linear_total.linear1'],
                        'drop_model': [drop_small_w], 'drop_connect': True},    
                       {'model_name': 'DropBigW', 'num_classes': 6, 'add_layer': ['linear_total.linear1'],
                        'drop_model': [drop_big_w], 'drop_connect': True},   
                       {'model_name': 'DropSmallGd', 'num_classes': 6, 'add_layer': ['linear_total.linear1'],
                        'drop_model': [drop_small_gd], 'drop_connect': True},
                       {'model_name': 'DropBigGd', 'num_classes': 6, 'add_layer': ['linear_total.linear1'],
                        'drop_model': [drop_big_gd], 'drop_connect': True},
                       ]
Result = []

for config in tzip(range(len(hyper_param_configs))):
    if hyper_param_configs[config[0]]['model_name'] == "Dropout":
        model = AlexNet.AlexNet_dropout(**hyper_param_configs[config[0]])
    else:
        model = AlexNet.AlexNet(**hyper_param_configs[config[0]])
    model.to(DEVICE)
    model.weights_init()
    model.compiler(lr=LR, weight_decay=weight_decay, lr_scheduler_apply=apply_lr_scheduler, 
                   device=DEVICE, cosine=True, first_cycle_steps=RUN_EPOCHS, 
                   max_lr=LR, warmup_steps=warmup_steps, gamma=0.2)
    model.fit(RUN_EPOCHS, train_loader, valid_loader, test_loader, show_test_result=True)
    print(model._Eval_Score(save=False))
    Result.append(model)
    
get_result_table(Result)
        
