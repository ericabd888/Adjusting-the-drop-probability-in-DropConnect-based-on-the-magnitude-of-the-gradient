import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from pytorch_model_summary import summary
from torch.optim.lr_scheduler import _LRScheduler
import io
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from warmup_scheduler import GradualWarmupScheduler

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def get_gd_distribution(gd_dict, add_layer):
    
    '''
    get parameter's distribution, we splilt them to three layers, 
    up layer, apply Regularization Method layer and down layer
    '''
    params_set = []
    target_p = None
    other_p_t1 = None
    other_p_t2 = None
    total_p = None
    up_layer = True
    for p in gd_dict:
        if p.find("weight") >= 0:
            if total_p is None:
                total_p = gd_dict[p].view(-1)
            else:
                total_p = torch.cat((total_p, gd_dict[p].view(-1)))  
        for target_layer in add_layer:
            if p.find(target_layer+".weight") >= 0:
                up_layer = False
                if target_p is None:
                    target_p = gd_dict[p].view(-1)
                else:
                    target_p = torch.cat((target_p, gd_dict[p].view(-1)))
        if up_layer and p.find("weight") >= 0:
            if other_p_t1 is None:
                other_p_t1 = gd_dict[p].view(-1)
            else:
                other_p_t1 = torch.cat((other_p_t1, gd_dict[p].view(-1)))  
        elif p.find(target_layer+".weight") < 0 and p.find("weight") >= 0:
            if other_p_t2 is None:
                other_p_t2 = gd_dict[p].view(-1)
            else:
                other_p_t2 = torch.cat((other_p_t2, gd_dict[p].view(-1))) 
    if other_p_t1 is not None:
        params_set.append(other_p_t1)
    params_set.append(target_p)
    params_set.append(other_p_t2)
    params_set.append(total_p)
    return [p.cpu().data.numpy() for p in params_set]
    
    
class Models(nn.Module):
    def __init__(self, p=0.5, model_name="base_model"):
        super(Models, self).__init__()
        self.add_layer = None
        self.drop_model = None
        self.normal_drop = False
        self.drop_connect = False
        self.p = p
        self.model_name = model_name
        self.best_test_acc_val = -1
        self.best_valid_acc_val = -1
        self.get_gd_info = False
        self.gd_distribution_list = np.array([None,None,None,None])
        self.gd_thr = 0.004
        self.First = False
    def weights_init(self):
        """the weights of conv layer and fully connected layers 
        are both initilized with Xavier algorithm, In particular,
        we set the parameters to random values uniformly drawn from [-a, a]
        where a = sqrt(6 * (din + dout)), for batch normalization 
        layers, y=1, b=0, all bias initialized to 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def compiler(self, optim_type="sgd", lr=0.0001, weight_decay=1e-5, momentum=0.9, 
                 no_bias_decay=False, optimizer=None, loss=None, cosine=False, step_size=50, 
                 lr_scheduler_apply=False, lr_scheduler=None, device=None, first_cycle_steps=390, 
                 cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=0.5, 
                 warm_up_lr_apply=False, warmup_init_lr=1e-5, warmup_iter=12, mile_stone=[20]):
        '''
            complie your model hyperparameter settings
        '''
        
        self.device = device
        self.lr_scheduler_apply = lr_scheduler_apply
        self.warm_up = warm_up_lr_apply
        self.optimizer = self.set_optim(optim_type=optim_type, lr=lr, weight_decay=weight_decay, 
                                        momentum=momentum, no_bias_decay=no_bias_decay)
        self.loss = self.set_loss()
        self.warmup_iteration = warmup_iter
        initial_lr = lr
        warmup_initial_lr = warmup_init_lr
        if self.lr_scheduler_apply:
            self.lr_scheduler=self.set_lr_scheduler(self.optimizer, cosine=cosine, step_size=step_size,
                                                    first_cycle_steps=first_cycle_steps, cycle_mult=cycle_mult, 
                                                    max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, 
                                                    gamma=gamma, mile_stone=mile_stone)
            if self.warm_up:       
                print("Warm up apply\n")
                self.lr_scheduler_warmup = create_lr_scheduler_with_warmup(self.lr_scheduler,
                                                               warmup_start_value=warmup_initial_lr,
                                                               warmup_duration=self.warmup_iteration,
                                                               warmup_end_value=initial_lr)
        
    def set_optim(self, optim_type="sgd", no_bias_decay=False, momentum=0.9, lr=1e-3, weight_decay=5e-5):
        '''
            set your optimizer
        '''
        params = split_weights(self) if no_bias_decay else self.parameters()
        if optim_type == "sgd":
            print("Adopt sgd optim\n")
            return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            print("Adopt Adam optim\n")
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    def set_loss(self):
        '''
            choose your loss function
        '''
        return nn.CrossEntropyLoss()
    
    def set_lr_scheduler(self, 
                         optimizer, 
                         cosine=False,
                         first_cycle_steps=200, 
                         cycle_mult=1.0, 
                         max_lr=0.1,
                         min_lr=0.001,
                         warmup_steps=50,
                         gamma=0.5,
                         step_size=50,
                         mile_stone=[80, 160, 240, 320, 400],
                         ):
        '''
        choose learning scheduler
        '''
        if cosine:
            print("Consine Apply\n")
#             return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=first_cycle_steps)
            return CosineAnnealingWarmupRestarts(optimizer, 
                                                 first_cycle_steps=first_cycle_steps, 
                                                 cycle_mult=cycle_mult, 
                                                 max_lr=max_lr, 
                                                 min_lr=min_lr, 
                                                 warmup_steps=warmup_steps, 
                                                 gamma=gamma)
        else:
            print("Step LR apply\n")
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_stone, gamma=gamma)
        
    def forward(self, x):
        '''
            forward pass, inherit it!
        '''
        return x
    def fit(self, EPOCHS, train_loader, valid_loader, test_loader, show_test_result=False, drop_layer=2):
        '''
            follow keras-like train mode, fit function, here is for training your model
        '''
        self.epoch = int(EPOCHS)
        print("Start Training...\nRunning {} .... ".format(self.model_name))
        self.lr_list = []
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []
        self.drop_rate_list = [[] for i in range(drop_layer)]
        print("Model Size: {}".format(sum(dict((p.data_ptr(), p.numel()) for p in self.parameters()).values())))
        if self.drop_connect:       
            print("Drop Connect Apply\n")
        
        for epoch in tqdm(range(1, self.epoch+1)):
            self.train()
            running_loss = 0
            correct = 0
            total = 0
            if self.warm_up:
                self.lr_scheduler_warmup(None)
            for i, data in enumerate(train_loader):
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if i == 0 and epoch == 1:
                    self.input_shape = inputs
                if self.drop_connect:
                    total_original_params = []
                    for drop_layer_idx in range(len(self.add_layer)):
                        temp_param = []
                        children_num_list = self.add_layer[drop_layer_idx].split(".")
                        temp_attr = self
                        for son in children_num_list:
                            temp_attr = getattr(temp_attr, son)
                        for n, p in temp_attr.named_parameters():
                            if(n=='weight'):
                                temp_param.append(p.clone())
                                with torch.no_grad():
                                    if self.normal_drop: 
                                        p.copy_(F.dropout(p, p=self.p))
                                    else:
                                        if i == 0 and epoch == 1:
                                            print("GradDrop Apply in {} layer\n".format(self.add_layer[drop_layer_idx]))
                                        p.copy_(self.drop_model[drop_layer_idx](p.grad, p, training=self.train())) 
                                        self.drop_rate_list[drop_layer_idx].append(
                                            [self.drop_model[drop_layer_idx].final_drop_rate,
                                             self.drop_model[drop_layer_idx].final_left_rate])
                        total_original_params.append(temp_param)
                    

                self.optimizer.zero_grad()

                outputs = self(inputs)

                
                loss = self.loss(outputs, labels)
                loss.backward()
                if self.drop_connect:
                    for drop_layer_idx in range(len(self.add_layer)):
                        children_num_list = self.add_layer[drop_layer_idx].split(".")
                        temp_attr = self
                        for son in children_num_list:
                            temp_attr = getattr(temp_attr, son)
                        for orig_p, (n, p) in zip(total_original_params[drop_layer_idx], temp_attr.named_parameters()):
                            if n == 'weight':
                                with torch.no_grad():
                                    if i == 0 and epoch == 1:
                                        print("Copy Back to {}\n".format(self.add_layer[drop_layer_idx]))
                                    p.copy_(orig_p)
                self.lr_list.append(self.optimizer.param_groups[0]['lr'])
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.detach(), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
 # Gradient Dictionary in abs
            if self.get_gd_info:
                if self.add_layer:
                    grad_dict = {k:torch.abs(v.grad) for k,v in self.named_parameters()}
                    final_gd_distribution = get_gd_distribution(grad_dict, self.add_layer)
                    for idx in range(len(final_gd_distribution)):
                        if self.gd_distribution_list[idx] is None:
                            self.gd_distribution_list[idx] = np.asarray([final_gd_distribution[idx]])
                        else:
                            self.gd_distribution_list[idx] = np.vstack([self.gd_distribution_list[idx], final_gd_distribution[idx]])

            running_loss /= len(train_loader)
            self.train_loss.append(running_loss)
            running_acc = correct / total
            self.train_acc.append(running_acc)
            
            valid_running_acc, valid_running_loss = self.validate(valid_loader)
            self.valid_acc.append(valid_running_acc)
            self.valid_loss.append(valid_running_loss)
            
            # record test best acc value
            if valid_running_acc >= self.best_valid_acc_val:
                print("Update New Weight\n")
                self.best_valid_acc_val = valid_running_acc
                self.model_best_weight = self.state_dict()
                
            if show_test_result:
                test_running_acc, test_running_loss = self.test(test_loader)
                
#                 # record test best acc value
#                 if test_running_acc >= self.best_test_acc_val:
#                     print("Update New Weight\n")
#                     self.best_test_acc_val = test_running_acc
#                     self.model_best_weight = self.state_dict()
                    
                self.test_acc.append(test_running_acc)
                self.test_loss.append(test_running_loss)
            
            if self.lr_scheduler_apply:
                if self.warm_up == False:
                    self.lr_scheduler.step()
            if epoch % 1 == 0 or epoch == EPOCHS:
                # print(self.drop_model[0].final_left_rate)
                print("\nCurrent Epoch: {}, LR: {}, Max Test Acc: {:.4f}, LeftRate: {}".format(epoch, self.optimizer.param_groups[0]['lr'], 100*max(self.test_acc), self.drop_model[0].final_left_rate if self.drop_model else None))
                print("Train Acc => {:.4f}".format(100 * running_acc), end=' | ')
                print("Train Loss => {:.4f}".format(running_loss))
                print("Valid Acc => {:.4f}".format(100 * valid_running_acc), end=' | ')
                print("Valid Loss => {:.4f}".format(valid_running_loss))
                if show_test_result:
                    print("Test Acc => {:.4f}".format(100 * test_running_acc), end=' | ')
                    print("Test Loss => {:.4f}".format(test_running_loss))
        self.best_acc_idx = np.argmax(self.valid_acc)
        self.best_loss_idx = np.argmin(self.valid_loss)
        self.best_test_idx = np.argmax(self.test_acc)
        self.plot_drop_layer_weight = []
        for drop_layer_idx in range(len(self.add_layer)):
            children_num_list = self.add_layer[drop_layer_idx].split(".")
            temp_attr = self
            for son in children_num_list:
                temp_attr = getattr(temp_attr, son)
            for n, p in temp_attr.named_parameters():
                if(n=='weight'):
                    with torch.no_grad():
                        self.plot_drop_layer_weight.append(p.view(-1).cpu().data.numpy())
        print("\nFinish Training\n")
        self.model_final_weight = self.state_dict()
        self.model_info = (self.train_acc, self.train_loss, self.valid_acc, self.valid_loss, self.test_acc, self.test_loss, self.best_acc_idx, self.best_loss_idx, self.best_test_idx) if show_test_result else (self.train_acc, self.train_loss, self.valid_acc, self.valid_loss, self.best_acc_idx, self.best_loss_idx)
        
    def validate(self, valid_loader):
        '''
            validate your validation set
        '''
        with torch.no_grad():
            correct = 0
            total = 0
            valid_running_loss = 0
            self.eval()
            for data in valid_loader:
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                
                loss = self.loss(outputs, labels)
                valid_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_running_loss /= len(valid_loader)
        valid_running_acc = correct / total
        return (valid_running_acc, valid_running_loss) 
    def test(self, test_loader):
        '''
            test your testing set
        '''
        with torch.no_grad():
            correct = 0
            total = 0
            test_running_loss = 0
            self.eval()
            for data in test_loader:
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                
                loss = self.loss(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_running_loss /= len(test_loader)
        test_running_acc = correct / total
        return (test_running_acc, test_running_loss)
    def _Eval_Score(self, picture_location=None, plot=True, show_input=False, show_hierarchical=False, save=True):
        '''
            After finish your training model, you can see your model training performance
        '''
        output = io.StringIO()
        print("Here is Model {}'s Result\n".format(self.model_name))
        print("Input Shape: ", self.input_shape.size(), file=output)
        print(summary(self.to(self.device), torch.zeros_like(self.input_shape), show_input=show_input, show_hierarchical=show_hierarchical), file=output)
        print("\n\nUse Final Epoch's Model to choose Model\n", file=output)
        print("Train acc: {:.4f}, Train Loss: {:.4f}".format(self.train_acc[-1], self.train_loss[-1]), file=output)
        print("Valid acc: {:.4f}, Valid Loss: {:.4f}".format(self.valid_acc[-1], self.valid_loss[-1]), file=output)
        print("Test acc: {:.4f}, Test Loss: {:.4f}".format(self.test_acc[-1], self.test_loss[-1]), file=output)
        print("\nUse Best Valid Loss to choose Model\n", file=output)
        print("Train acc: {:.4f}, Train Loss: {:.4f}".format(self.train_acc[self.best_loss_idx], self.train_loss[self.best_loss_idx]), file=output)
        print("Valid acc: {:.4f}, Valid Loss: {:.4f}".format(self.valid_acc[self.best_loss_idx], self.valid_loss[self.best_loss_idx]), file=output)
        print("Test acc: {:.4f}, Test Loss: {:.4f}".format(self.test_acc[self.best_loss_idx], self.test_loss[self.best_loss_idx]), file=output)
        print("\nUse Best Valid Acc to choose Model\n", file=output)
        print("Train acc: {:.4f}, Train Loss: {:.4f}".format(self.train_acc[self.best_acc_idx], self.train_loss[self.best_acc_idx]), file=output)
        print("Valid acc: {:.4f}, Valid Loss: {:.4f}".format(self.valid_acc[self.best_acc_idx], self.valid_loss[self.best_acc_idx]), file=output)
        print("Test acc: {:.4f}, Test Loss: {:.4f}".format(self.test_acc[self.best_acc_idx], self.test_loss[self.best_acc_idx]), file=output)
        print("\nUse Best Test Acc to choose Model\n", file=output)
        print("Train acc: {:.4f}, Train Loss: {:.4f}".format(self.train_acc[self.best_test_idx], self.train_loss[self.best_test_idx]), file=output)
        print("Valid acc: {:.4f}, Valid Loss: {:.4f}".format(self.valid_acc[self.best_test_idx], self.valid_loss[self.best_test_idx]), file=output)
        print("Test acc: {:.4f}, Test Loss: {:.4f}".format(self.test_acc[self.best_test_idx], self.test_loss[self.best_test_idx]), file=output)
        contents = output.getvalue()
        output.close()
        if plot:
            self._plot(location=picture_location, save=save)
        
        return contents
    def _plot(self, location=None, save=True):
        '''
            plot training result
        '''
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        picture_name = location if location else (self.model_name+'.png')
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("train, valid, test", fontsize=16)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        ax1.plot(range(1, self.epoch+1), self.train_acc, label='train')
        ax1.plot(range(1, self.epoch+1), self.valid_acc, label='validation')
        ax1.plot(range(1, self.epoch+1), self.test_acc, label='test')
        ax1.set_title("Accuracy")
        ax1.legend()
        
        ax2.plot(range(1, self.epoch+1), self.train_loss, label='train')
        ax2.plot(range(1, self.epoch+1), self.valid_loss, label='validation')
        ax2.plot(range(1, self.epoch+1), self.test_loss, label='test')
        ax2.set_title("Loss")
        ax2.legend()     
        
        ax3.plot(range(1, len(self.lr_list)+1), self.lr_list, label='total_lr')
        ax3.set_title("total learning rate")
        ax3.legend()

        if self.drop_model:
            plot_drop_list = [np.array(i) for i in self.drop_rate_list]
            for idx in range(len(self.add_layer)):
                ax4.plot(range(1, plot_drop_list[idx].shape[0]+1), plot_drop_list[idx][:, 1].flatten(), 
                         label='LeftRate', c=color_list[idx], linestyle='--')
            ax4.set_title("Drop Rate and Left Rate")
            ax4.legend()
        else:
            fig.delaxes(ax4)
        if save:
            plt.savefig(picture_name)
        plt.show()
    def _save(self, FILE=None, transform_dict=None, ):
        '''
            choose your file name and save it!
        '''
#         FILE = "./Model_pool/simple_cnn_table5_vallina_aug3.pt"
        state = {
            'epoch' : self.epoch,
            'model_name' : self.model_name,
            'train_aug' : str(transform_dict['train']),
            'valid_aug' : str(transform_dict['valid']),
            'test_aug' : str(transform_dict['test']),
            'state_dict' : self.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'model_quick_look' : self._Eval_Score(),
            'model_info' : self.model_info,
            'drop_model': [model.model_hyper_params for model in self.drop_model] if self.drop_model else None
        }
        torch.save(state, FILE)

''' 
---------DEMO------------
class model_edit(Models):
    # if you don't need to change base model's model, your can use it directly.
    def do_something():
        pass
class model_test(Models):
    # if you need to modify base model function method, you should add super()
    def __init__(self, vgg_name, num_classes):
        super(model_test, self)._make_layers(cfg[vgg_name])
        super(model_test, self).__init__(vgg_name, num_classes)
    def fit(self, EPOCHS, train_loader, test_loader, criterion, optimizer, device):
        return super(model_test, self).fit(EPOCHS, train_loader, test_loader, criterion, optimizer, device) 
    def test(self, test_loader, criterion, optimizer, device):
        return super(model_test, self).test(test_loader, criterion, optimizer, device)
    def forward(self, x):
        return super(model_test, self).forward(x)

## Here is the most correct use way

you should declare your __init__(), forward(), init_weights() functions, and 
check the base_model's fit function can work well on your model


class model_drop_connect(Models):
    def __init__(self, num_classes, add_layer=None, drop_model=None, drop_connect=False):
        super(model_drop_connect, self).__init__()
        self.add_layer = add_layer
        self.drop_model = drop_model
        self.drop_connect = drop_connect
        self.name = self.drop_model
        self.conv_layers = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            nn.Conv2d(64, 192, kernel_size=5, padding=2),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            nn.Conv2d(192, 384, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(384, 256, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            )
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, num_classes)
        self.init_weights()
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    def init_weights(self):
        for layer in self.conv_layers.children():
            if hasattr(layer, "reset_parameters"):
                print("layer initialize : ", layer)
                layer.reset_parameters()
        reset_list = [self.fc1, self.fc2, self.fc3]
        for m in reset_list:
            print("layer initialize: ", m)
            m.reset_parameters()
'''


