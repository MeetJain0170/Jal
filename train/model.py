import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    """
    Advanced Transfer-Learning UNet using a ResNet34 backbone explicitly pre-trained on ImageNet.
    Replaces the custom scratch-built Attention R2U-Net to achieve "amazing" quality on sparse data.
    """
    def __init__(self, in_channels=3, out_channels=3, base=64):
        super(UNet, self).__init__()
        # Ignore "base", keep it for backward compatibility with model_loader.py calls
        
        print(">> INITIALIZING PRE-TRAINED RESNET34 U-NET (TRANSFER LEARNING) <<")
        # Initialize the state-of-the-art UNet from smp
        self.model = smp.Unet(
            encoder_name="resnet34",        # Use ResNet34 backbone
            encoder_weights="imagenet",     # Initialize with ImageNet pre-trained weights
            in_channels=in_channels,
            classes=out_channels,
            activation=None                 # Raw logits
        )

    def forward(self, x):
        # Forward pass
        out = self.model(x)
        # Apply sigmoid to constraint output between 0 and 1
        out = torch.sigmoid(out)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)