# Training a neural network by adjusting the drop probability in DropConnect based on the magnitude of the gradient
## Introduction
A Gradient DropConnect Model by adjusting the drop probability based on the magnitude of the gradient.

## QuickStart

### SimpleCNN1_MNIST_Exp.py
* Training SimpleCNN1 Network on MNIST, with Vallina, Dropout, DropConnect, DropSmallW, DropBigW, DropSmallGd, DropBigBd technique.
### Usage:
```
$ python SimpleCNN1_MNIST_Exp.py
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
$ python SimpleCNN1_MNIST_Exp.py
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
