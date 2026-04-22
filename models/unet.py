#unet.py
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two conv layers with batch norm and ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder: conv + downsample"""
    
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv_out = self.conv(x)
        pooled = self.pool(conv_out)
        return conv_out, pooled


class DecoderBlock(nn.Module):
    """Decoder: upsample + concat skip + conv"""
    
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net for underwater image enhancement
    
    Encoder extracts features at multiple scales
    Decoder reconstructs enhanced image using skip connections
    """
    
    def __init__(self, in_channels=3, out_channels=3, init_features=64):
        super(UNet, self).__init__()
        
        f = init_features
        
        # encoder path
        self.encoder1 = EncoderBlock(in_channels, f)
        self.encoder2 = EncoderBlock(f, f * 2)
        self.encoder3 = EncoderBlock(f * 2, f * 4)
        self.encoder4 = EncoderBlock(f * 4, f * 8)
        
        # bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16)
        
        # decoder path
        self.decoder1 = DecoderBlock(f * 16, f * 8)
        self.decoder2 = DecoderBlock(f * 8, f * 4)
        self.decoder3 = DecoderBlock(f * 4, f * 2)
        self.decoder4 = DecoderBlock(f * 2, f)
        
        # output layer
        self.final = nn.Conv2d(f, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # encoder
        s1, x = self.encoder1(x)
        s2, x = self.encoder2(x)
        s3, x = self.encoder3(x)
        s4, x = self.encoder4(x)
        
        x = self.bottleneck(x)
        
        # decoder with skip connections
        x = self.decoder1(x, s4)
        x = self.decoder2(x, s3)
        x = self.decoder3(x, s2)
        x = self.decoder4(x, s1)
        
        x = self.final(x)
        x = self.sigmoid(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
