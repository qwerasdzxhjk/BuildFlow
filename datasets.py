import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
import random
import einops


def center_crop_arr(pil_image, image_size):
    """Center crop image to specified size."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


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

        source_image = Image.open(source_path).convert("RGB")
        target_image_pil = Image.open(target_path).convert("L") # Load as a single-channel grayscale image

        source_image_arr = np.array(source_image)
        target_mask_arr = np.array(target_image_pil)
        target_mask_arr = target_mask_arr * 255
        
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
    def __init__(self, latent_root, target_root):
        self.latent_root = latent_root
        self.target_root = target_root
        self.latent_files = sorted(glob.glob(os.path.join(latent_root, '*.npz')))
        print(f"Found {len(self.latent_files)} preprocessed latent pairs")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = self.latent_files[idx]
        
        # --- Fix 1: safely read npz files with a context manager ---
        # Using 'with' ensures file handles are released correctly and avoids worker hangs or read errors
        with np.load(latent_path) as data:
            # --- Fix 2: use .clone() ---
            # np.load returns a memory-mapped view, and converting it directly to Tensor may pin memory
            # clone() copies the data into memory and disconnects it from the backing file
            source_latent = torch.from_numpy(data['source_latent']).float().clone()
            target_latent = torch.from_numpy(data['target_latent']).float().clone()

        # Process the mask while adding an existence check
        mask_fname = os.path.basename(latent_path).replace('.npz', '.tif')
        mask_path = os.path.join(self.target_root, mask_fname)

        # --- Fix 3: add a file existence check to avoid crashes ---
        if not os.path.exists(mask_path):
            # If the corresponding mask is missing, create an all-black mask to avoid crashing
            # This often happens when dataset preprocessing is incomplete
            print(f"Warning: Mask not found for {mask_fname}, using black mask.")
            # The default shape is assumed to be 256x256; adjust it if your actual data shape differs
            target_mask_tensor = torch.zeros((1, 256, 256), dtype=torch.float32) 
        else:
            try:
                target_mask_pil = Image.open(mask_path).convert("L")
                target_mask_arr = np.array(target_mask_pil)
                # Normalize and add the channel dimension
                target_mask_tensor = torch.from_numpy(target_mask_arr / 255.0).float()
                target_mask_tensor = target_mask_tensor.unsqueeze(0) # [H, W] -> [1, H, W]
            except Exception as e:
                print(f"Error reading mask {mask_path}: {e}")
                target_mask_tensor = torch.zeros((1, 256, 256), dtype=torch.float32)

        return {
            'source_latent': source_latent,
            'target_latent': target_latent,
            'target_masks': target_mask_tensor,
            'filename': os.path.basename(latent_path)
        }

class Image2IDataFactory:
    def __init__(self, config):
        self.config = config
        self.image_size = config.dataset.image_size
        self.batch_size = config.train.batch_size
        self.num_workers = config.train.num_workers

    def create_train_dataloader(self, use_preprocessed=True):
        if use_preprocessed:
            dataset = PreprocessedImagePairDataset(
                latent_root=self.config.dataset.train_latent_root,
                target_root=self.config.dataset.train_target_root
            )
        else:
            dataset = ImagePairDataset(
                source_root=self.config.dataset.train_source_root,
                target_root=self.config.dataset.train_target_root,
                image_size=self.image_size
            )

        # --- Fix 4: handle the strategy for the final incomplete batch ---
        # Set this to False if you want to keep all samples, including the final partial batch
        # If that causes downstream shape mismatches, the model code must be updated or this must stay True
        drop_last = True 
        
        # Deadlocks can happen when persistent_workers=True does not release workers cleanly at the end of reading
        # If the dataset is small, forcing this to False is usually safer
        use_persistent_workers = True if len(dataset) > self.batch_size * 10 else False

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=use_persistent_workers, 
            drop_last=drop_last 
        )

    def create_val_dataloader(self, use_preprocessed=True):
        """Create validation data loader."""
        if use_preprocessed:
            dataset = PreprocessedImagePairDataset(
                latent_root=self.config.dataset.val_latent_root,
                target_root=self.config.dataset.val_target_root # Keep behavior consistent for the validation set
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
            num_workers=self.num_workers,
            pin_memory=True
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
