# Training a neural network by adjusting the drop probability in DropConnect based on the magnitude of the gradient
## Introduction
A Gradient DropConnect Model by adjusting the drop probability based on the magnitude of the gradient.

## SetUp
Tested under Python 3.8.12 in Ubuntu. Install the required packages by
```
$ pip install -r requirements.txt
```
## QuickStart
## MNIST
### SimpleCNN1_MNIST_Exp.py
* Training SimpleCNN1 Network on MNIST, with Vallina, Dropout, DropConnect, DropSmallW, DropBigW, DropSmallGd, DropBigBd technique.
### Usage:
```
$ python exp/MNIST_exp/SimpleCNN1_MNIST_Exp.py
      --epoch <number of epoch>
      --lr <learning rate>
      --weight_decay <optimizer weight decay>
      --dropout_rate <Dropout Drop Rate>
      --dropconnect_rate <DropConnect Drop Rate>
      --gd_init_droprate <Drop Small/Big Gd Init Drop Rate>
      --gd_droprate <Drop Small/Big Gd based on Gradient Drop Rate>
      --w_init_droprate <Drop Small/Big Weight Init Drop Rate>
      --w_droprate <Drop Small/Big Weight based on Weight Value Drop Rate>
      --batch_size <Batch Size>
      --apply_lr_scheduler <Apply Learning Rate Scheduler>
      --warmup_steps <We use warmup plus Learning Rate Scheduler, epoch number must bigger than warmup_steps, if not will occur error>
      --no-apply_lr_scheduler <This setting can cancel warmup and learning rate scheduler>
```
### Default Example:
```
$ python exp/MNIST_exp/SimpleCNN1_MNIST_Exp.py
      --epoch 75
      --lr 0.001
      --weight_decay 5e-4
      --dropout_rate 0.5
      --dropconnect_rate 0.5
      --gd_init_droprate 0.4
      --gd_droprate 0.25
      --w_init_droprate 0.4
      --w_droprate 0.25
      --batch_size 128
      --apply_lr_scheduler 
      --warmup_steps 12
```
## CIFAR10
### SimpleCNN1_CIFAR10_Exp.py
* Training SimpleCNN1 Network on CIFAR10, with Vallina, Dropout, DropConnect, DropSmallW, DropBigW, DropSmallGd, DropBigBd technique.
### Default Example:
```
$ python exp/CIFAR10_exp/SimpleCNN1_CIFAR10_Exp.py
      --epoch 120
      --lr 0.005
      --weight_decay 5e-4
      --dropout_rate 0.5
      --dropconnect_rate 0.5
      --gd_init_droprate 0.4
      --gd_droprate 0.25
      --w_init_droprate 0.4
      --w_droprate 0.25
      --batch_size 128
      --apply_lr_scheduler 
      --warmup_steps 12
```
### SimpleCNN2_CIFAR10_Exp.py
* Training SimpleCNN2 Network on CIFAR10, with Vallina, Dropout, DropConnect, DropSmallW, DropBigW, DropSmallGd, DropBigBd technique.
### Default Example:
```
$ python exp/CIFAR10_exp/SimpleCNN2_CIFAR10_Exp.py
      --epoch 130
      --lr 0.055
      --weight_decay 5e-5
      --dropout_rate 0.5
      --dropconnect_rate 0.5
      --gd_init_droprate 0.4
      --gd_droprate 0.25
      --w_init_droprate 0.4
      --w_droprate 0.25
      --batch_size 128
      --apply_lr_scheduler 
      --warmup_steps 12
```
### AlexNet_CIFAR10_Exp.py
* Training AlexNet Network on CIFAR10, with Vallina, Dropout, DropConnect, DropSmallW, DropBigW, DropSmallGd, DropBigBd technique.
### Default Example:
```
$ python exp/CIFAR10_exp/AlexNet_CIFAR10_Exp.py
      --epoch 40
      --lr 0.01
      --weight_decay 5e-4
      --dropout_rate 0.5
      --dropconnect_rate 0.5
      --gd_init_droprate 0.4
      --gd_droprate 0.25
      --w_init_droprate 0.4
      --w_droprate 0.25
      --batch_size 128
      --apply_lr_scheduler 
      --warmup_steps 8
```
### VGG_CIFAR10_Exp.py
* Training VGG Network on CIFAR10, with Vallina, Dropout, DropConnect, DropSmallW, DropBigW, DropSmallGd, DropBigBd technique.
### Default Example:
```
$ python exp/CIFAR10_exp/VGG_CIFAR10_Exp.py
      --epoch 100
      --lr 0.005
      --weight_decay 0
      --dropout_rate 0.5
      --dropconnect_rate 0.5
      --gd_init_droprate 0.4
      --gd_droprate 0.25
      --w_init_droprate 0.4
      --w_droprate 0.25
      --batch_size 128
      --apply_lr_scheduler 
      --warmup_steps 10
```
## CIFAR100
### VGG_CIFAR10_Exp.py
* Training VGG Network on CIFAR100, with Vallina, Dropout, DropConnect, DropSmallW, DropBigW, DropSmallGd, DropBigBd technique.
### Default Example:
```
$ python exp/CIFAR100_exp/VGG_CIFAR100_Exp.py
      --epoch 220
      --lr 0.06
      --weight_decay 8e-4
      --dropout_rate 0.5
      --dropconnect_rate 0.5
      --gd_init_droprate 0.4
      --gd_droprate 0.25
      --w_init_droprate 0.4
      --w_droprate 0.25
      --batch_size 128
      --apply_lr_scheduler 
      --warmup_steps 20
```
## NORB
### SimpleCNN2_NORB_Exp.py
* Training SimpleCNN2 Network on NORB, with Vallina, Dropout, DropConnect, DropSmallW, DropBigW, DropSmallGd, DropBigBd technique.
### Default Example:
```
$ python exp/NORB_exp/SimpleCNN2_NORB_Exp.py
      --epoch 100
      --lr 0.001
      --weight_decay 6.5e-3
      --dropout_rate 0.5
      --dropconnect_rate 0.5
      --gd_init_droprate 0.4
      --gd_droprate 0.25
      --w_init_droprate 0.4
      --w_droprate 0.25
      --batch_size 128
      --apply_lr_scheduler 
      --warmup_steps 10
```
### AlexNet_NORB_Exp.py
* Training AlexNet Network on NORB, with Vallina, Dropout, DropConnect, DropSmallW, DropBigW, DropSmallGd, DropBigBd technique.
### Default Example:
```
$ python exp/NORB_exp/AlexNet_NORB_Exp.py
      --epoch 75
      --lr 0.0002
      --weight_decay 0
      --dropout_rate 0.5
      --dropconnect_rate 0.5
      --gd_init_droprate 0.4
      --gd_droprate 0.25
      --w_init_droprate 0.4
      --w_droprate 0.25
      --batch_size 128
      --apply_lr_scheduler 
      --warmup_steps 8
```
### VGG_NORB_Exp.py
* Training VGG Network on NORB, with Vallina, Dropout, DropConnect, DropSmallW, DropBigW, DropSmallGd, DropBigBd technique.
### Default Example:
```
$ python exp/NORB_exp/VGG_NORB_Exp.py
      --epoch 85
      --lr 0.0008
      --weight_decay 4e-4
      --dropout_rate 0.5
      --dropconnect_rate 0.5
      --gd_init_droprate 0.4
      --gd_droprate 0.25
      --w_init_droprate 0.4
      --w_droprate 0.25
      --batch_size 128
      --apply_lr_scheduler 
      --warmup_steps 10
```
