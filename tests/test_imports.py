import importlib


def test_can_import_models():
    modules = [
        "models.gradient_dropconnect",
        "models.base_model",
        "models.AlexNet_cifar10",
        "models.AlexNet_norb",
        "models.SimpleCNN1_cifar10",
        "models.SimpleCNN1_mnist",
        "models.SimpleCNN2_cifar10",
        "models.SimpleCNN2_norb",
        "models.VGG_cifar10",
        "models.VGG_cifar100",
        "models.VGG_norb",
    ]
    for name in modules:
        importlib.import_module(name)
