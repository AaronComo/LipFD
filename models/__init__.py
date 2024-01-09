from .clip_models import CLIPModel
from .LipFD import LipFD, RALoss

VALID_NAMES = [
    "CLIP:ViT-B/32",
    "CLIP:ViT-B/16",
    "CLIP:ViT-L/14",
]


def get_model(name):
    assert name in VALID_NAMES
    if name.startswith("CLIP:"):
        return CLIPModel(name[5:])
    else:
        assert False


def build_model(transformer_name):
    assert transformer_name in VALID_NAMES
    if transformer_name.startswith("CLIP:"):
        return LipFD(transformer_name[5:])
    else:
        assert False


def get_loss():
    return RALoss()
