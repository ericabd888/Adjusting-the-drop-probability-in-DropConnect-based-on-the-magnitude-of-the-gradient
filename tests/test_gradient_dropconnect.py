import sys
import types
import torch

# Mock heavy optional deps before importing module
sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))
sys.modules.setdefault("torchvision.transforms", types.ModuleType("torchvision.transforms"))

from models.gradient_dropconnect import GradWeightDrop


def test_gradweightdrop_forward_pass_no_grad_returns_weight():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = GradWeightDrop(device=device, I_P=0.5, W_P=0.0, GD_P=0.0)
    w = torch.randn(4, 4, device=device)

    # First call (grad is None or cur_time==0) should return weight unchanged
    out = layer(grad=None, weight=w, training=True)
    assert torch.allclose(out, w)


def test_gradweightdrop_mask_shape_matches_weight():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure final_drop_p becomes a tensor by enabling weight-based path
    layer = GradWeightDrop(device=device, I_P=0.2, W_P=0.0, GD_P=0.0, w_small=False)
    w = torch.randn(8, 8, device=device)
    g = torch.randn_like(w)

    # Warm up once so cur_time > 0
    _ = layer(grad=None, weight=w, training=True)

    out = layer(grad=g, weight=w, training=True)
    assert out.shape == w.shape
