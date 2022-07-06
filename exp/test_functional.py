import sys
sys.path.insert(1, '../')
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
import torchvision

import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils.helper import cal_std_mean_and_get_plot_data, save_std_mean, plot_all_model, compute_mean_std
from base_model import Models
from gradient_dropconnect import GradWeightDrop as gd2
from torch.nn.modules.utils import _pair
import random
import matplotlib.pyplot as plt
from random import shuffle
import pickle
from collections import OrderedDict

from torch.backends import cudnn

cudnn.benchmark = True # fast training

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("n", help="repeat time", type=int)
parser.add_argument("-u", "--user-name", dest="user_name")

args = parser.parse_args()
for _ in range(args.n):
    print("Hello, {}".format(args.user_name))

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform


    def __getitem__(self, index):
        x, y = self.subset[index]
#         y = y.type(torch.LongTensor)
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    

BATCH_SIZE = 128




train_set = torchvision.datasets.MNIST(root='../data', train=True,
                                        download=True)

test_set = torchvision.datasets.MNIST(root='../data', train=False,
                                       download=True)

total_data_len = len(train_set)
target = [i for i in range(total_data_len)]
shuffle(target)
train_idx = target[:54000]
valid_idx = target[54000:]



training_set = torch.utils.data.Subset(train_set, train_idx)
validating_set = torch.utils.data.Subset(train_set, valid_idx)


mean1, std1 = compute_mean_std(training_set)
mean2, std2 = compute_mean_std(validating_set)
mean3, std3 = compute_mean_std(test_set)


total_transform = {
    'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean1, std1)
            ]),
    'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean2, std2),
            ]),
    'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean3, std3),
            ]),
} 


training_set = MyDataset(training_set, transform=total_transform['train'])
validating_set = MyDataset(validating_set, transform=total_transform['valid'])
testing_set = MyDataset(test_set, transform=total_transform['test'])

train_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)

valid_loader = DataLoader(validating_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)


test_loader = DataLoader(testing_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

print(mean1, std1)
print(mean2, std2)
print(mean3, std3)
print(len(training_set))
print(len(validating_set))
print(len(testing_set))



class Reshape_layer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)
class basic_linear(Models):
    def __init__(self, vgg_name, num_classes, add_layer=None, 
                 drop_model=None, drop_connect=False, apply_lr_scr=True, 
                 normal_drop=False, p=0.5, model_name="123"):
        super().__init__()
        self.linear_total = nn.Sequential(OrderedDict([
            ('flattern', nn.Flatten()),
            ('linear1', nn.Linear(784, 1500)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(1500, 800)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(800, 10)),
        ]))
        self.model_name = model_name
        self.name = vgg_name
        self.add_layer = add_layer
        self.drop_model = drop_model
        self.drop_connect = drop_connect
        self.normal_drop = normal_drop
        self.p = p
    def forward(self, x):
        x = self.linear_total(x)
        return x

class basic_cnn(Models):
    def __init__(self, vgg_name, num_classes, add_layer=None, 
                 drop_model=None, drop_connect=False, apply_lr_scr=True, 
                 normal_drop=False, p=0.5, model_name="123"):
        super().__init__()
        self.cnn = nn.Sequential(OrderedDict([
            ('flattern1', Reshape_layer(-1,28,28)),
            ('cnn1', nn.Conv2d(1, 3, 3)),
            ('maxpool1', nn.MaxPool2d((2,2))),
            ('cnn2', nn.Conv2d(3, 6, 3)),
            ('maxpool2', nn.MaxPool2d((2,2))),
        ]))
        self.linear_total = nn.Sequential(OrderedDict([
            ('flattern', nn.Flatten()),
            ('linear1', nn.Linear(150, 600)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(600, 200)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(200, 10)),
        ]))
        self.model_name = model_name
        self.name = vgg_name
        self.add_layer = add_layer
        self.drop_model = drop_model
        self.drop_connect = drop_connect
        self.normal_drop = normal_drop
        self.p = p
    def forward(self, x):
        x = self.cnn(x)
        x = self.linear_total(x)
        return x



LR = 0.0006
weight_decay = 0
mini_lr = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)




# Add Regularization Method
grad_drop1_1 = gd2(DEVICE, I_P=0.0, W_P=0., GD_P=0.75, w_small=True, gd_small=True, name="First")


grad_drop2_1 = gd2(DEVICE, I_P=0.0, W_P=0., GD_P=1.29, w_small=True, gd_small=True, name="First")


grad_drop3_1 = gd2(DEVICE, I_P=0.3, W_P=0., GD_P=0.25, w_small=True, gd_small=True, name='DropConnectFirst')


grad_drop4_1 = gd2(DEVICE, I_P=0.4, W_P=0., GD_P=0.35, w_small=True, gd_small=True, name="First")


grad_drop5_1 = gd2(DEVICE, I_P=0.3, W_P=0., GD_P=0.5, w_small=True, gd_small=True, name="First")


grad_drop6_1 = gd2(DEVICE, I_P=0.54, W_P=0., GD_P=0.0, w_small=True, gd_small=True, name="First")


setting_dict_list = [
            {'vgg_name':'VGG11', 'num_classes':10, 'add_layer':['linear_total.linear1'],
              'model_name':'Vallina'},
            {'vgg_name':'VGG11', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
              'drop_model':[grad_drop1_1], 'drop_connect':True, 'model_name':'Pure_Gd_Drop_40'},
            {'vgg_name':'VGG11', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
              'drop_model':[grad_drop2_1], 'drop_connect':True, 'model_name':'Pure_Gd_Drop_54'},
            {'vgg_name':'VGG11', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
              'drop_model':[grad_drop3_1], 'drop_connect':True, 'model_name':'GD_40'},
            {'vgg_name':'VGG11', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
              'drop_model':[grad_drop4_1], 'drop_connect':True, 'model_name':'GD_54'},
            {'vgg_name':'VGG11', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
              'drop_model':[grad_drop5_1], 'drop_connect':True, 'model_name':'DropConnect_gd54+'},
            {'vgg_name':'VGG11', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
              'drop_model':[grad_drop6_1], 'drop_connect':True, 'model_name':'DropConnect_54'},
            {'vgg_name':'VGG11', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
             'drop_connect':True, 'normal_drop':True, 'model_name':'B_DropConnect_40', 'p':0.3},
            {'vgg_name':'VGG11', 'num_classes':10, 'add_layer':['linear_total.linear1'], 
             'drop_connect':True, 'normal_drop':True, 'model_name':'B_DropConnect_60', 'p':0.5},
            ]




# fig = plt.figure(figsize=(20,20))
# ax1 = fig.add_subplot(2,2,1)
# ax2 = fig.add_subplot(2,2,2)
# ax3 = fig.add_subplot(2,2,3)
# ax4 = fig.add_subplot(2,2,4)

# import pandas as pd
# eps = 1e-10
# p_value_test = []
# for i in run_model:
#     model = basic_cnn(**setting_dict_list[i])
#     model.to(DEVICE)
#     model.weights_init()
#     model.compiler(lr=LR, weight_decay=weight_decay, lr_scheduler_apply=True, 
#                                 device=DEVICE, cosine=True, max_lr=LR, first_cycle_steps=60, 
#                     warmup_steps=0, gamma=0.2, warm_up_lr_apply=True, warmup_iter=12)
#     model.fit(20, train_loader, valid_loader, test_loader, show_test_result=True)
#     show_1 = model.gd_distribution_list
#     print(show_1)
#     N = 5
#     width = 0.3
#     color = ['b', 'g', 'r', 'c', 'm']
#     X_AXIS = np.arange(1,20+1)
#     index = pd.Index(X_AXIS, name="up")
#     dis_name = ["16", "32", "48", "64", "80"]
#     data = [{},{},{},{}]
#     for i in range(5):
#         data[0][dis_name[i]] = show_1[0][:,i]
#         data[1][dis_name[i]] = show_1[1][:,i]
#         data[2][dis_name[i]] = show_1[2][:,i]
#         data[3][dis_name[i]] = show_1[3][:,i]
        
#     df1 = pd.DataFrame(data[0], index=index)
#     df1.plot(ax=ax1, kind='bar', stacked=True)
#     df2 = pd.DataFrame(data[1], index=index)
#     df2.plot(ax=ax2, kind='bar', stacked=True)
#     df3 = pd.DataFrame(data[2], index=index)
#     df3.plot(ax=ax3, kind='bar', stacked=True)
#     df4 = pd.DataFrame(data[3], index=index)
#     df4.plot(ax=ax4, kind='bar', stacked=True)
    
# #     for i in range(5):
# #         ax1.bar(ind, show_1[0][:,i], width, bottom=show_1[0][:,i-1] if i!=0 else 0, color=color[i])
# #         ax2.bar(ind, show_1[1][:,i], width, bottom=show_1[1][:,i-1] if i!=0 else 0, color=color[i])
# #         ax3.bar(ind, show_1[2][:,i], width, bottom=show_1[2][:,i-1] if i!=0 else 0, color=color[i])
# #         ax4.bar(ind, show_1[3][:,i], width, bottom=show_1[3][:,i-1] if i!=0 else 0, color=color[i])
#     ax1.set_title("Up")
#     h, label = ax1.get_legend_handles_labels()
#     ax1.legend(reversed(h), reversed(label), title='labels', bbox_to_anchor=(1.0,1), loc='upper left')
    
#     ax2.set_title("Apply")
#     h, label = ax2.get_legend_handles_labels()
#     ax2.legend(reversed(h), reversed(label),title='labels', bbox_to_anchor=(1.0,1), loc='upper left')
# #     ax3.bar(range(show_1[0].shape[0]), show_1[2].T)
#     ax3.set_title("Down")
#     h, label = ax3.get_legend_handles_labels()
#     ax3.legend(reversed(h), reversed(label),title='labels', bbox_to_anchor=(1.0,1), loc='upper left')
# #     ax4.bar(range(show_1[0].shape[0]), show_1[3].T)
#     ax4.set_title("All")
#     h, label = ax4.get_legend_handles_labels()
#     ax4.legend(reversed(h), reversed(label),title='labels', bbox_to_anchor=(1.0,1), loc='upper left')
    
    
    
# #     ax1.plot(range(len(show_1[0])), show_1[0], label=i)
# #     ax1.set_title("Up")
# #     ax1.legend()
# #     ax2.plot(range(len(show_1[1])), show_1[1], label=i)
# #     ax2.set_title("Apply")
# #     ax2.legend()
# #     ax3.plot(range(len(show_1[2])), show_1[2], label=i)
# #     ax3.set_title("Down")
# #     ax3.legend()
# #     ax4.plot(range(len(show_1[3])), show_1[3], label=i)
# #     ax4.set_title("total")
# #     ax4.legend()
# fig.savefig("hihi.png")


def split_model_weight_distribution(model):
    params_set = []
    print(len(model.model_best_weight))
    target_p = None
    other_p_t1 = None
    other_p_t2 = None
    up_layer =  True
    for p in model.model_best_weight:
        for target_layer in model.add_layer:
            if p.find(target_layer+".weight") >= 0:
                up_layer = False
                print("Target Layer Size: ", model.model_best_weight[p].size())
                if target_p is None:
                    target_p = model.model_best_weight[p].view(-1)
                else:
                    target_p = torch.cat((target_p, model.model_best_weight[p].view(-1)))
        if up_layer and p.find("weight") >= 0:
            print("Up Layer Size: ", model.model_best_weight[p].size())
            if other_p_t1 is None:
                other_p_t1 = model.model_best_weight[p].view(-1)
            else:
                other_p_t1 = torch.cat((other_p_t1, model.model_best_weight[p].view(-1)))  
        elif p.find(target_layer+".weight") < 0 and p.find("weight") >= 0:
            print("Down Layer Size: ", model.model_best_weight[p].size())
            if other_p_t2 is None:
                other_p_t2 = model.model_best_weight[p].view(-1)
            else:
                other_p_t2 = torch.cat((other_p_t2, model.model_best_weight[p].view(-1))) 
                
    if other_p_t1 is not None:
        params_set.append(other_p_t1)
    params_set.append(target_p)
    params_set.append(other_p_t2)
    return [p.cpu().data.numpy() for p in params_set]


print("Start Test training correct or not")
model = basic_linear(**setting_dict_list[4])
model.to(DEVICE)
model.weights_init()
model.compiler(lr=LR, weight_decay=weight_decay, lr_scheduler_apply=True, 
                            device=DEVICE, cosine=True, max_lr=LR, first_cycle_steps=60, 
                warmup_steps=0, gamma=0.2, warm_up_lr_apply=True, warmup_iter=12)

model.fit(10, train_loader, valid_loader, test_loader, show_test_result=True)
print(model._Eval_Score(plot=True))
print("Finish this time")
# Exp_time = 1
# RUN_EPOCH = 75

# compute_all = []


# FILE = "./Model_pool/04_13"
# # MODEL = "/cifar100_vgg16_vallina
# # pic_FILE = FILE + MODEL + ".png"
# # model_FILE = FILE + MODEL + ".pt"


# plot_row_num = len(setting_dict_list)
# plot_col_num = 3
# fig, axs = plt.subplots(plot_row_num, plot_col_num, figsize=(plot_col_num*5, 4*plot_row_num))
# print("ax size: ", len(axs))
# fig.suptitle('Weight Distribution in Mnist', fontsize=12)
# axs = axs.ravel()
# three_layers_name = ["Up", "Target", "Down"]
# # p_ax = []
# # plot_fig = plt.figure(figsize=(12, 12))
# # p_ax.append(plot_fig.add_subplot(2, 2, 1))
# # p_ax.append(plot_fig.add_subplot(2, 2, 2))
# # p_ax.append(plot_fig.add_subplot(2, 2, 3))
# # p_ax.append(plot_fig.add_subplot(2, 2, 4))

# model_name = []

# xtick_list = [[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 1.0, 1.5], 
#               [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
#               [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]]

# ytick_list = [[30, 5], [800, 100], [7500, 500]]

# x_lim_list = [[1.5], [0.3], [0.3]]

# y_lim_list = [[25], [5000], [7000]]

# for m in tzip(range(len(setting_dict_list))):
#     temp_list = []
#     for time in tqdm(range(Exp_time)):
#         model = basic_cnn(**setting_dict_list[m[0]])
#         model.to(DEVICE)
#         model.weights_init()
#         model.compiler(lr=LR, weight_decay=weight_decay, lr_scheduler_apply=True, 
#                                     device=DEVICE, cosine=True, max_lr=LR, first_cycle_steps=75, 
#                         warmup_steps=0, gamma=0.2, warm_up_lr_apply=True, warmup_iter=12)
#         model.fit(RUN_EPOCH, train_loader, valid_loader, test_loader, show_test_result=True)
#         if model.drop_model:
#             for drop in model.drop_model:
#                 drop._reset_time()
#         if time == 0:
# #             print(model._Eval_Score(save=False))
#             params_set = split_model_weight_distribution(model)
#             for p_idx in range(len(params_set)):
#                 axs[m[0]*3+p_idx].hist(params_set[p_idx], bins=75, color="blue")
                
#                 axs[m[0]*3+p_idx].set_xticks(xtick_list[p_idx])
#                 axs[m[0]*3+p_idx].set_yticks(np.arange(0, ytick_list[p_idx][0], ytick_list[p_idx][1]))
#                 axs[m[0]*3+p_idx].set_xlim(-1*x_lim_list[p_idx][0], x_lim_list[p_idx][0])
#                 axs[m[0]*3+p_idx].set_ylim(0, y_lim_list[p_idx][0])
#                 axs[m[0]*3+p_idx].set_title(model.model_name+" "+three_layers_name[p_idx]+" Weight Dis", fontsize=9)
#             model_name.append(model.model_name)
#             print(model._Eval_Score(picture_location=FILE+'/CompareDropMnist_linear_exp_'+model.model_name+'.png'))
#             model._save(FILE+'/CompareDropMnist_linear_exp_'+model.model_name+'.pt', total_transform)
#         temp_list.append(model.model_info)
#     compute_all.append(temp_list)



# fig.savefig(FILE+'/CompareDropMnist_linear_distribution.pdf')

# final_result = cal_std_mean_and_get_plot_data(compute_all, model_name)
# save_std_mean(final_result, FILE+"/CompareDropMnist_linear_exp_mean_std.npy")


# plot_all_model(FILE+'/CompareDropMnist_linear_all_model_combine.png', final_result, model_name)