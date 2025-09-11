import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Dataset

class GradWeightDrop(nn.Module):
    def __init__(self, device, I_P=0.5, W_P=0, GD_P=0, w_small=None, gd_small=None, name="DropConnectModel"):
        super().__init__()
        self.eps = 1e-9
        self.I_P = I_P # Each Weight Initial Drop Rate
        self.W_P = W_P # Consider Each Weight Value Generate Max Drop Rate
        self.GD_P = GD_P # Consider Each Gradient Value Generate Max Drop Rate
        self.device = device
        self.w_small = w_small # Drop Big or Small Weight
        self.gd_small = gd_small # Drop Big or Small Gradient
        self.cur_time = 0
        self.final_drop_rate = 0 # For Inverted Dropout, need consider total drop rate
        self.final_left_rate = 1 # For Inverted Dropout, need consider total left rate
        self.drop_name = name
        self.THR = 0.5
        self.model_hyper_params = {
            "Init Prob": self.I_P,
            "Weight Prob": self.W_P,
            "Gradient Prob": self.GD_P,
            "w_small": self.w_small,
            "gd_small": self.gd_small,
        }
    def forward(self, grad, weight, training=True):
        if grad == None or self.cur_time == 0:
            self.cur_time += 1
            return weight
        else:
            if training:
                # get weight and gradient absolute value

                final_drop_p = self.I_P
                if self.w_small is not None:
                    weight_abs = torch.abs(weight)
                
                
                    w_mean = torch.mean(weight_abs)
                    w_std = torch.std(weight_abs)
                    standard_weight = (weight_abs - w_mean) / (w_std + self.eps)

                    # w_sigmoid high mean w is big
                    w_sigmoid = torch.sigmoid(standard_weight)
                    if self.w_small:
                        w_sigmoid = 1 - w_sigmoid

                    '''
                    add some test
                    '''
                    thr_w_sigmoid = (w_sigmoid > self.THR)
                    w_sigmoid = thr_w_sigmoid * w_sigmoid
                    final_drop_p += self.W_P*w_sigmoid
                if self.gd_small is not None:

                
                    grad_abs = torch.abs(grad)
                    gd_mean = torch.mean(grad_abs)
                    gd_std = torch.std(grad_abs)



                    standard_gd = (grad_abs - gd_mean) / (gd_std + self.eps) #z-score
                    gd_sigmoid = torch.sigmoid(standard_gd)
                
                    if self.gd_small:
                        gd_sigmoid = 1 - gd_sigmoid
                
                    '''
                    add some test
                    '''

                    thr_gd_sigmoid = (gd_sigmoid > self.THR)
                    gd_sigmoid = thr_gd_sigmoid * gd_sigmoid
        
                    final_drop_p += self.GD_P*gd_sigmoid

                final_mask = self._mask(final_drop_p)


                left_rate = torch.sum(final_mask) / final_mask.view(-1).size(0)
                divide_drop_p_value = torch.sum(final_drop_p) / final_drop_p.view(-1).size(0)
                self.final_drop_rate = divide_drop_p_value.cpu().data.numpy()
                self.final_left_rate = left_rate.cpu().data.numpy()
                if self.cur_time == 1:
                    print("GradDrop Final Drop rate: ", divide_drop_p_value)
                    print("GradDrop Final Left rate: ", left_rate)
                    
                self.cur_time += 1
#                 return (weight) / (divide_drop_p_value + self.eps)
#                 return (weight * final_mask)/ ((1 - divide_drop_p_value) + self.eps)
                return (weight * final_mask)  / (self.final_left_rate + self.eps)
            else:
                self.cur_time += 1
                return weight
    def _mask(self, p):
        r"""Given a Tensor of probabilities 'p' this function
           will sample a mask nd return the mask (the probabilities are retention probabilities)
           
                   P                uniform_P
                [0.1, 0.3],      [0.3, 0.2],
                [0.9, 0.65]  >   [0.7, 0.9]     """
        #Random Sampling
        uniform = torch.cuda.FloatTensor(p.size()).uniform_(0, 1).to(self.device) if torch.cuda.is_available() else torch.FloatTensor(p.size()).uniform_(0, 1)
        #Setting Mask
        mask = p >= uniform
        mask = torch.logical_not(mask)
        #Setting proper Data Type
        return mask
    def _reset_time(self):
        self.cur_time = 0
        self.final_drop_rate = 0
        self.final_left_rate = 1
        
