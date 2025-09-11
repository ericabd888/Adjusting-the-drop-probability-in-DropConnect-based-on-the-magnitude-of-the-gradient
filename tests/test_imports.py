import importlib


def test_can_import_core_models():
    modules = [
        "models.gradient_dropconnect",
        "models.base_model",
    ]
    for name in modules:
        importlib.import_module(name)
