"""Pre-encode image/mask pairs into VAE latents for faster training.

Edit the paths in main(), then run:

    python extract_features.py

Every matched image/mask pair (same filename stem, .tif) produces one .npz
archive holding the source and target latents consumed by the training
config's train_latent_root / val_latent_root.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get(
    'BUILD_FLOW_CUDA_VISIBLE_DEVICES',
    os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
)
import shutil

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import einops
import libs.autoencoder
from glob import glob


def recreate_folder(folder_path):
    """Remove and recreate folder."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def extract_i2i_features_from_folders(source_root, target_root, save_dir,
                                    vae_path, resolution=256, batch_size=16, device="cuda"):
    """
    Extract I2I features from source and target image folders.
    
    ## MODIFIED TO MATCH ImagePairDataset LOGIC ##
    """

    recreate_folder(save_dir)

    autoencoder = libs.autoencoder.get_model(vae_path, scale_factor=0.23010)
    autoencoder.to(device)
    autoencoder.eval()

    # --- MODIFICATION 1: Find *.tif files instead of *.jpg ---
    source_files = {os.path.basename(f): f for f in glob(os.path.join(source_root, '*.tif'))}
    target_files = {os.path.basename(f): f for f in glob(os.path.join(target_root, '*.tif'))}

    common_files = sorted(list(set(source_files.keys()).intersection(set(target_files.keys()))))

    print(f"Found {len(common_files)} matching image pairs")

    if len(common_files) == 0:
        print("No matching image pairs found!")
        print(f"Source files: {len(source_files)}")
        print(f"Target files: {len(target_files)}")
        return

    for i in tqdm(range(0, len(common_files), batch_size), desc="Extracting features"):
        batch_files = common_files[i:i + batch_size]
        batch_source_imgs, batch_target_imgs, batch_names = [], [], []

        for filename in batch_files:
            try:
                # --- SOURCE IMAGE PROCESSING (matches ImagePairDataset) ---
                source_pil = Image.open(source_files[filename]).convert("RGB")
                # --- MODIFICATION 2: Removed center_crop_arr for source ---
                source_img = np.array(source_pil)
                source_img = (source_img / 127.5 - 1.0).astype(np.float32)
                source_img = einops.rearrange(source_img, 'h w c -> c h w')
                batch_source_imgs.append(source_img)

                # --- TARGET IMAGE PROCESSING (matches ImagePairDataset) ---
                # --- MODIFICATION 3: Replicated the exact target processing logic ---
                target_pil = Image.open(target_files[filename]).convert("L")
                target_img_arr = np.array(target_pil)

                # Normalize masks to a stable binary 0/255 representation.
                # This handles both 0/1 and 0/255 source masks without uint8 overflow.
                target_img_arr = (target_img_arr > 0).astype(np.uint8) * 255

                target_img = np.stack([target_img_arr] * 3, axis=-1) # Stack to 3 channels
                
                target_img = (target_img / 127.5 - 1.0).astype(np.float32)
                target_img = einops.rearrange(target_img, 'h w c -> c h w')
                batch_target_imgs.append(target_img)

                batch_names.append(filename.replace('.tif', '')) # Changed to .tif

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

        if len(batch_source_imgs) == 0:
            continue

        # Convert to tensors
        batch_source_tensor = torch.tensor(np.stack(batch_source_imgs)).to(device)
        batch_target_tensor = torch.tensor(np.stack(batch_target_imgs)).to(device)

        with torch.no_grad():
            # The VAE encoding part remains the same
            # Note: `resolution` parameter is no longer used for cropping but might be
            # needed by the VAE if it has different behaviors. The code here is generic.
            # Keep the leading batch dimension.  Calling squeeze(0) here
            # corrupts a final batch of size one by turning ``[1, C, H, W]``
            # into ``[C, H, W]`` and making the loop below iterate channels.
            source_latent = autoencoder(batch_source_tensor, fn='encode_moments')
            target_latent = autoencoder(batch_target_tensor, fn='encode_moments')

        if source_latent.shape[0] != len(batch_names):
            raise RuntimeError(
                f"VAE returned {source_latent.shape[0]} samples for "
                f"{len(batch_names)} input files."
            )

        # Save features
        for j, (name, source_lat, target_lat) in enumerate(zip(batch_names, source_latent, target_latent)):
            save_path = os.path.join(save_dir, f'{name}.npz')
            data = {
                'source_latent': source_lat.cpu().numpy(),
                'target_latent': target_lat.cpu().numpy(),
                'source_name': name,
                'target_name': name,
                # We save the nominal resolution, even if not used for cropping
                'resolution': resolution 
            }
            np.savez(save_path, **data)

    print(f"Feature extraction completed. Saved to {save_dir}")


def main():
    """Main function for extracting I2I features."""

    # Configuration - modify these paths according to your dataset
    config = {
        # TODO: Fill in your local VAE checkpoint path before feature extraction.
        'vae_path': '',
        # Option 1: Automatic pairing by filename
        # TODO: Fill in your local source image directory before feature extraction.
        'source_root': '',
        # TODO: Fill in your local target mask directory before feature extraction.
        'target_root': '',
        # TODO: Fill in your local latent output directory before feature extraction.
        'save_dir': '',
        'resolution': 512,
        'batch_size': 8,
        'device': 'cuda',
    }

    print("Starting I2I feature extraction...")
    print(f"Config: {config}")


    # Use automatic filename pairing
    extract_i2i_features_from_folders(
        source_root=config['source_root'],
        target_root=config['target_root'],
        save_dir=config['save_dir'],
        vae_path=config['vae_path'],
        resolution=config['resolution'],
        batch_size=config['batch_size'],
        device=config['device']
        )


if __name__ == '__main__':
    main()
