import torch
import os
import sys
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import math

import config
from train.model import UNet

def load_trained_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    
    model = UNet(
        in_channels=config.IMAGE_CHANNELS,
        out_channels=config.IMAGE_CHANNELS,
        base=config.UNET_INIT_FEATURES
    ).to(config.DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def enhance_patch(model, patch_tensor):
    """Helper to run model on a single patch"""
    with torch.no_grad():
        out = model(patch_tensor)
        # 🛡️ Safety Blend (20% original structure retention)
        # This keeps sharp edges of divers/mines from getting "dreamed" away
        out = 0.8 * out + 0.2 * patch_tensor
        return torch.clamp(out, 0, 1)

def enhance_high_res(model, image_path, output_path, patch_size=256, stride=256):
    """
    Splits high-res image into patches, enhances them, and stitches back.
    PRESERVES ORIGINAL RESOLUTION.
    """
    print(f"Processing high-res image: {image_path}...")
    
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    print(f"Original Resolution: {w}x{h}")

    # Convert to tensor (C, H, W)
    full_transform = transforms.ToTensor()
    img_tensor = full_transform(img).to(config.DEVICE)
    
    # Create empty output tensor
    output_tensor = torch.zeros_like(img_tensor)
    count_map = torch.zeros((1, h, w), device=config.DEVICE)

    # Calculate padding to fit patch size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    # Pad image if necessary
    if pad_h > 0 or pad_w > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        output_tensor = torch.nn.functional.pad(output_tensor, (0, pad_w, 0, pad_h))
        count_map = torch.nn.functional.pad(count_map, (0, pad_w, 0, pad_h))
        
    padded_h, padded_w = img_tensor.shape[1], img_tensor.shape[2]

    # Sliding Window Inference
    for y in range(0, padded_h - patch_size + 1, stride):
        for x in range(0, padded_w - patch_size + 1, stride):
            # Extract patch
            patch = img_tensor[:, y:y+patch_size, x:x+patch_size].unsqueeze(0)
            
            # Enhance
            enhanced_patch = enhance_patch(model, patch)
            
            # Add to output
            output_tensor[:, y:y+patch_size, x:x+patch_size] += enhanced_patch.squeeze(0)
            count_map[:, y:y+patch_size, x:x+patch_size] += 1.0

    # Handle right/bottom edges if stride doesn't cover perfectly (simple approach: crop back)
    # Average overlapping regions
    output_tensor = output_tensor / torch.clamp(count_map, min=1.0)
    
    # Crop back to original size
    output_tensor = output_tensor[:, :h, :w]
    
    # Save
    out_np = output_tensor.cpu().permute(1, 2, 0).numpy()
    out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(out_np)
    
    out_img.save(output_path, quality=100) # Save as max quality
    print(f"Saved High-Res Output: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py input.jpg output.jpg")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'enhanced_output.png'
    
    try:
        model = load_trained_model()
        if os.path.isfile(input_path):
            enhance_high_res(model, input_path, output_path)
        else:
            print(f"Error: {input_path} not found")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()