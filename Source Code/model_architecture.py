import torch
import segmentation_models_pytorch as smp

def get_model(device="cuda"):
    """
    Initializes and returns the UnetPlusPlus model with EfficientNet-b3 encoder.
    """
    model = smp.UnetPlusPlus(
        encoder_name    = "efficientnet-b3",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = None
    )
    return model.to(device)