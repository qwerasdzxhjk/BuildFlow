"""Datasets and data loaders for BuildFlow training and validation."""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import numpy as np
import einops


class ImagePairDataset(Dataset):
    """
    Dataset for image-to-image pairs.
    Loads source and target image pairs for training I2I flow matching.
    """

    def __init__(self,
                 source_root,
                 target_root,
                 image_size=256,
                 pair_file=None):
        """
        Args:
            source_root: Directory containing source images
            target_root: Directory containing target images
            image_size: Target image size for resizing/cropping
            pair_file: Optional file containing image pair mappings
        """
        self.source_root = source_root
        self.target_root = target_root
        self.image_size = image_size

        # Get image file mappings
        if pair_file:
            # Load predefined pairs from file
            with open(pair_file, 'r') as f:
                self.pairs = [line.strip().split(',') for line in f.readlines()]
        else:
            # Automatic pairing by filename
            source_files = set(os.path.basename(f) for f in glob.glob(os.path.join(source_root, '*.tif')))
            target_files = set(os.path.basename(f) for f in glob.glob(os.path.join(target_root, '*.tif')))

            # Find common files
            common_files = source_files.intersection(target_files)
            self.pairs = [(f, f) for f in common_files]

            print(f"Found {len(self.pairs)} matching image pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_name, target_name = self.pairs[idx]

        # Load images
        source_path = os.path.join(self.source_root, source_name)
        target_path = os.path.join(self.target_root, target_name)

        with Image.open(source_path) as source_image:
            source_image_arr = np.array(source_image.convert("RGB"))
        with Image.open(target_path) as target_image_pil:
            target_mask_arr = np.array(target_image_pil.convert("L"))

        # Normalize masks to a stable binary 0/255 representation.
        # This handles both 0/1 and 0/255 source masks without uint8 overflow.
        target_mask_arr = (target_mask_arr > 0).astype(np.uint8) * 255
        
        # --- Prepare target_images for flow matching ---
        # Replicate the single-channel mask into 3 channels to match the VAE input
        target_image_for_fm = np.stack([target_mask_arr] * 3, axis=-1)
        
        # Normalize images to [-1, 1]
        source_tensor = (source_image_arr / 127.5 - 1.0).astype(np.float32)
        target_tensor = (target_image_for_fm / 127.5 - 1.0).astype(np.float32)
        
        source_tensor = einops.rearrange(source_tensor, 'h w c -> c h w')
        target_tensor = einops.rearrange(target_tensor, 'h w c -> c h w')

        # --- Prepare target_masks for the segmentation loss ---
        # Normalize mask values from [0, 255] to [0.0, 1.0] and keep float32 dtype
        target_mask_tensor = (target_mask_arr / 255.0).astype(np.float32)
        # Add a channel dimension so the shape becomes [1, H, W]
        target_mask_tensor = einops.rearrange(target_mask_tensor, 'h w -> 1 h w')

        return {
            'source_images': torch.from_numpy(source_tensor),
            'target_images': torch.from_numpy(target_tensor),
            'target_masks': torch.from_numpy(target_mask_tensor), # Additional returned item
            'source_names': source_name,
            'target_names': target_name
        }


class PreprocessedImagePairDataset(Dataset):
    """Pre-encoded VAE latents produced by extract_features.py."""

    def __init__(self, latent_root):
        self.latent_root = latent_root
        self.latent_files = sorted(glob.glob(os.path.join(latent_root, '*.npz')))
        print(f"Found {len(self.latent_files)} preprocessed latent pairs")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = self.latent_files[idx]

        # Read npz files with a context manager so file handles are released
        # correctly; clone() copies the data into memory, disconnecting it from
        # the backing file.
        with np.load(latent_path) as data:
            source_latent = torch.from_numpy(data['source_latent']).float().clone()
            target_latent = torch.from_numpy(data['target_latent']).float().clone()

        return {
            'source_latent': source_latent,
            'target_latent': target_latent,
        }

class Image2IDataFactory:
    def __init__(self, config):
        self.config = config
        self.image_size = config.dataset.image_size
        self.batch_size = config.train.batch_size

    def create_train_dataloader(self, use_preprocessed=True):
        if use_preprocessed:
            dataset = PreprocessedImagePairDataset(
                latent_root=self.config.dataset.train_latent_root
            )
        else:
            dataset = ImagePairDataset(
                source_root=self.config.dataset.train_source_root,
                target_root=self.config.dataset.train_target_root,
                image_size=self.image_size
            )

        # Data loading runs in the main process, matching the paper training
        # runs; the latent caches are small npz archives, so worker processes
        # are unnecessary.
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

    def create_val_dataloader(self, use_preprocessed=True):
        """Create validation data loader."""
        if use_preprocessed:
            dataset = PreprocessedImagePairDataset(
                latent_root=self.config.dataset.val_latent_root
            )
        else:
            dataset = ImagePairDataset(
                source_root=self.config.dataset.val_source_root,
                target_root=self.config.dataset.val_target_root,
                image_size=self.image_size
            )

        return DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )

    def unpreprocess(self, v):
        """Convert normalized tensor back to [0, 1] range."""
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v


def create_i2i_dataloaders(config):
    """Convenience function to create I2I data loaders."""
    factory = Image2IDataFactory(config)

    train_loader = factory.create_train_dataloader(use_preprocessed=config.use_preprocessed)
    # --- Critical fix: the validation loader must use raw images for visualization ---
    val_loader = factory.create_val_dataloader(use_preprocessed=False)

    return train_loader, val_loader, factory
