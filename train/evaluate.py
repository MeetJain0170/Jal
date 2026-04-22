#evaluate.py
import torch
import os
import numpy as np
from tqdm import tqdm

import config
from train.model import UNet
from dataset import UnderwaterDataset
from utils import evaluate_batch, visualize_results


def evaluate_model(model_path=None):
    """Evaluate trained model on validation set"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60 + "\n")
    
    if model_path is None:
        model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Train the model first: python train.py")
        return
    
    print(f"Loading model from: {model_path}")
    
    model = UNet(
        in_channels=config.IMAGE_CHANNELS,
        out_channels=config.IMAGE_CHANNELS,
        init_features=config.UNET_INIT_FEATURES
    ).to(config.DEVICE)
    
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model from epoch {checkpoint['epoch']}")
    
    # load validation data
    print(f"\nLoading validation dataset...")
    
    full_dataset = UnderwaterDataset(
        raw_dir=config.RAW_DIR,
        enhanced_dir=config.ENHANCED_DIR,
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        augment=False
    )
    
    train_size = int(config.TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    _, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    print(f"Validation set: {val_size} images\n")
    
    # run evaluation
    print("Running evaluation...")
    
    all_psnr = []
    all_ssim = []
    
    with torch.no_grad():
        for raw_imgs, target_imgs in tqdm(val_loader):
            raw_imgs = raw_imgs.to(config.DEVICE)
            target_imgs = target_imgs.to(config.DEVICE)
            
            preds = model(raw_imgs)
            
            batch_psnr, batch_ssim = evaluate_batch(preds, target_imgs)
            all_psnr.append(batch_psnr)
            all_ssim.append(batch_ssim)
    
    # results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    avg_psnr = np.mean(all_psnr)
    std_psnr = np.std(all_psnr)
    avg_ssim = np.mean(all_ssim)
    std_ssim = np.std(all_ssim)
    
    print(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.3f} ± {std_ssim:.3f}")
    print(f"PSNR range: [{np.min(all_psnr):.2f}, {np.max(all_psnr):.2f}] dB")
    print(f"SSIM range: [{np.min(all_ssim):.3f}, {np.max(all_ssim):.3f}]")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    evaluate_model()
