#dataset.py
import os
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class UnderwaterDataset(Dataset):
    """
    Paired dataset: degraded underwater image -> clean reference image.
    Images are resized and mapped to [0, 1]. Augmentations keep pairs in sync.
    """

    def __init__(self, raw_dir: str, enhanced_dir: str,
                 image_size: Tuple[int, int] = (256, 256),
                 augment: bool = False):
        super().__init__()

        self.raw_dir = raw_dir
        self.enhanced_dir = enhanced_dir
        self.image_size = image_size
        self.augment = augment

        self.image_files = self._get_paired_images()
        print(f"Found {len(self.image_files)} paired images")

        self.to_tensor = transforms.ToTensor()

    def _get_paired_images(self):
        raw_files = set(os.listdir(self.raw_dir))
        enhanced_files = set(os.listdir(self.enhanced_dir))
        paired = sorted(raw_files & enhanced_files)

        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        paired = [f for f in paired if os.path.splitext(f)[1].lower() in valid_ext]
        paired = [f for f in paired if not f.startswith('.')]
        return paired

    def _apply_augment(self, img_raw: Image.Image, img_ref: Image.Image):
        """Apply the same random flip/rotation/color jitter to both images."""
        # random flips
        if random.random() < 0.5:
            img_raw = TF.hflip(img_raw)
            img_ref = TF.hflip(img_ref)
        if random.random() < 0.3:
            img_raw = TF.vflip(img_raw)
            img_ref = TF.vflip(img_ref)

        # slight rotation (-5 to 5 degrees)
        angle = random.uniform(-5.0, 5.0)
        img_raw = TF.rotate(img_raw, angle, interpolation=TF.InterpolationMode.BILINEAR)
        img_ref = TF.rotate(img_ref, angle, interpolation=TF.InterpolationMode.BILINEAR)

        # low-intensity color jitter
        b = random.uniform(0.9, 1.1)
        c = random.uniform(0.9, 1.1)
        s = random.uniform(0.9, 1.1)
        h = random.uniform(-0.02, 0.02)
        img_raw = TF.adjust_brightness(img_raw, b)
        img_ref = TF.adjust_brightness(img_ref, b)
        img_raw = TF.adjust_contrast(img_raw, c)
        img_ref = TF.adjust_contrast(img_ref, c)
        img_raw = TF.adjust_saturation(img_raw, s)
        img_ref = TF.adjust_saturation(img_ref, s)
        img_raw = TF.adjust_hue(img_raw, h)
        img_ref = TF.adjust_hue(img_ref, h)

        return img_raw, img_ref

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        raw_path = os.path.join(self.raw_dir, filename)
        enhanced_path = os.path.join(self.enhanced_dir, filename)

        raw_img = Image.open(raw_path).convert('RGB')
        enhanced_img = Image.open(enhanced_path).convert('RGB')

        # resize first for consistent scale
        raw_img = raw_img.resize(self.image_size, Image.BILINEAR)
        enhanced_img = enhanced_img.resize(self.image_size, Image.BILINEAR)

        if self.augment:
            raw_img, enhanced_img = self._apply_augment(raw_img, enhanced_img)

        raw_tensor = self.to_tensor(raw_img)       # [0,1]
        enhanced_tensor = self.to_tensor(enhanced_img)

        return raw_tensor, enhanced_tensor


def get_dataloaders(raw_dir, enhanced_dir, batch_size, train_split=0.8,
                    image_size=(256, 256), num_workers=2, pin_memory=True):
    """
    Create train and validation dataloaders with a fixed random split.
    """
    full_dataset = UnderwaterDataset(
        raw_dir=raw_dir,
        enhanced_dir=enhanced_dir,
        image_size=image_size,
        augment=True
    )

    if len(full_dataset) == 0:
        raise ValueError("No images found. Check your data directories.")

    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # disable augmentation on validation set
    train_dataset.dataset.augment = True
    val_dataset.dataset.augment = False

    print(f"Split: {train_size} train, {val_size} val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader

