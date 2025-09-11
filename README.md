# Gradient DropConnect: Adjusting Drop Probability by Gradient Magnitude

Short description: Train CNNs with Gradient DropConnect, which adapts DropConnect probability based on the magnitude of parameter gradients. Includes MNIST, CIFAR-10/100 and NORB experiments with AlexNet, VGG, and SimpleCNN baselines.

## Features
- Gradient-aware DropConnect and baselines (Vanilla, Dropout, DropConnect, DropSmall/Big-W, DropSmall/Big-Gd)
- Ready-to-run experiment scripts per dataset/model
- Reproducible defaults, warmup + LR scheduler support

## Screenshots
Place images in `assets/screenshots/` and they will render here.

![Training curves](assets/screenshots/training_curves.png)
![CIFAR10 results table](assets/screenshots/cifar10_results.png)
![NORB sample batch](assets/screenshots/norb_samples.png)
![Gradient mask heatmap](assets/screenshots/gradient_mask_heatmap.png)

## Quickstart
Tested with Python 3.8.12.

```bash
pip install -r requirements.txt
```

Minimal runs (defaults can be overridden by flags below):

```bash
# MNIST (SimpleCNN1 + GradientDropConnect)
python exp/MNIST_exp/SimpleCNN1_MNIST_Exp.py --apply_lr_scheduler --warmup_steps 12

# CIFAR-10 (VGG + GradientDropConnect)
python exp/CIFAR10_exp/VGG_CIFAR10_Exp.py --apply_lr_scheduler --warmup_steps 8

# CIFAR-100 (VGG + GradientDropConnect)
python exp/CIFAR100_exp/VGG_CIFAR100_Exp.py --apply_lr_scheduler --warmup_steps 20

# NORB (SimpleCNN2 + GradientDropConnect)
python exp/NORB_exp/SimpleCNN2_NORB_Exp.py --apply_lr_scheduler --warmup_steps 10
```

Common flags supported by most scripts:

```bash
--epoch N --lr LR --weight_decay WD --dropout_rate P --dropconnect_rate P \
--gd_init_droprate P --gd_droprate P --w_init_droprate P --w_droprate P \
--batch_size N --apply_lr_scheduler --warmup_steps N
```

## Datasets
- NORB preprocessed tensors (`final_train_data.pt`, `final_test_data.pt`): [Download](https://drive.google.com/file/d/1i45E2uTEJPoNuXuGeRjiIvfNwBAd1S8I/view?usp=sharing)

Place the downloaded files under `data/big_norb_data/` if needed by your run.

## Topics
Suggested GitHub topics to improve discoverability:
- deep-learning
- regularization
- dropconnect
- pytorch
- computer-vision
- gradient-based-learning

You can set them with GitHub’s UI (Repo → Settings → General → Topics) or CLI:
`gh repo edit --add-topic deep-learning --add-topic regularization --add-topic dropconnect --add-topic pytorch --add-topic computer-vision --add-topic gradient-based-learning`

## Repo structure
```
exp/                  # Experiment entrypoints per dataset
models/               # Model definitions + Gradient DropConnect module
utils/                # Helpers
data/                 # Dataset assets (incl. NORB tensors)
assets/screenshots/   # Project screenshots referenced in README
```

## Reference
See `GradDropConnect_ijcai23(1).pdf` for the paper/preprint.

If you use this work, please cite:
```
@inproceedings{grad-dropconnect-ijcai23,
  title     = {Training a Neural Network by Adjusting Drop Probability in DropConnect Based on Gradient Magnitude},
  booktitle = {IJCAI},
  year      = {2023}
}
```

## License
MIT
