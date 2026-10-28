"""Reconstruction quality of the frozen VAE on binary mask images.

Encodes each ground-truth mask with the VAE and decodes it back (paper
Table XII), then reports IoU, F1, B-F1, B-SSIM, and B-IoU of the reconstructed
mask against the input.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get(
    'BUILD_FLOW_CUDA_VISIBLE_DEVICES',
    os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
)
import torch
import argparse
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm


from metrics import (
    boundary_iou,
    calculate_bf1_matlab,
    calculate_bssim_strict,
    calculate_f1_score,
    calculate_iou,
)
import libs.autoencoder

def load_ground_truth_mask(image_path: str):
    mask = Image.open(image_path).convert("L")
    mask_np = np.array(mask)
    mask_np = (mask_np > 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_np)
    return mask_tensor

def preprocess_image(image_path: str):
    image = Image.open(image_path).convert("L")
    image_tensor = np.array(image)
    # Binarize to a stable 0/1 representation (handles both 0/1 and 0/255
    # source masks without uint8 overflow) and map to [-1, 1].
    image_tensor = (image_tensor > 0).astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0)
    image_tensor = image_tensor.repeat(3, 1, 1)
    image_tensor = image_tensor * 2.0 - 1.0
    image_tensor_4d = image_tensor.unsqueeze(0)
    return image_tensor_4d

def encode_latents(vae: torch.nn.Module, x: torch.Tensor):
    x = x.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
    latent = vae.encode(x)
    return latent

def decode_latents(vae: torch.nn.Module, latents: torch.Tensor):
    latents = latents.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
    image = vae.decode(latents)
    image = (image / 2 + 0.5).clamp(0, 1).float()
    image = (image * 255.0).to(torch.uint8).cpu()
    return image

def main():
    parser = argparse.ArgumentParser(description="Test script for VAE with comprehensive metric calculation on a folder.")

    # TODO: Fill in your local VAE checkpoint path before running this script.
    parser.add_argument("--vae_path", type=str,
                        default="",
                        help="Path to the VAE model weights used in training.")
    parser.add_argument("--scale_factor", type=float, default=0.23010,
                        help="The scale factor for the VAE model, must match training configuration.")
    # TODO: Fill in your local mask directory before running this script.
    parser.add_argument("--mask_dir", type=str,
                        default="",
                        help="Path to the directory containing mask images for testing.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU to run.")
        return

    print("--- VAE Batch Testing Script (Calculating Multiple Metrics) ---")
    print(f"Loading VAE weights from: {args.vae_path}")
    print(f"Using VAE scale_factor: {args.scale_factor}")
    
    try:
        vae_config = {
            "pretrained_path": args.vae_path,
            "scale_factor": args.scale_factor
        }
        vae = libs.autoencoder.get_model(**vae_config)
        vae.requires_grad_(False)
        vae.eval()
        vae.cuda()
        print("VAE model loaded and configured successfully.")
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        return

    image_extensions = ["*.tif", "*.png", "*.jpg", "*.jpeg"]
    mask_files = []
    for ext in image_extensions:
        mask_files.extend(glob.glob(os.path.join(args.mask_dir, ext)))

    if not mask_files:
        print(f"No supported image files found in directory '{args.mask_dir}'.")
        return

    print(f"Found {len(mask_files)} images in the directory, starting processing...")

    all_iou_scores = []
    all_f1_scores = []
    all_bf1_scores = []
    all_bssim_scores = []
    all_biou_scores = []

    for i, mask_path in enumerate(tqdm(mask_files, desc="Processing")):
        try:
            input_tensor = preprocess_image(mask_path)

            ground_truth_tensor = load_ground_truth_mask(mask_path)
            gt_numpy = ground_truth_tensor.numpy().astype(np.uint8)

            with torch.no_grad():
                latents = encode_latents(vae, input_tensor)
                reconstructed_image_tensor = decode_latents(vae, latents)

            reconstructed_gray = reconstructed_image_tensor[0, 0, :, :].float()
            reconstructed_binary_tensor = (reconstructed_gray > 127.5).float()
            pred_numpy = reconstructed_binary_tensor.numpy().astype(np.uint8)

            iou_score = calculate_iou(reconstructed_binary_tensor, ground_truth_tensor)
            all_iou_scores.append(iou_score)

            f1_score = calculate_f1_score(reconstructed_binary_tensor, ground_truth_tensor)
            all_f1_scores.append(f1_score)

            bf1_score = calculate_bf1_matlab(pred_numpy, gt_numpy, distance_threshold=2.0)
            all_bf1_scores.append(bf1_score)

            bssim_score = calculate_bssim_strict(pred_numpy, gt_numpy)
            all_bssim_scores.append(bssim_score)

            biou_score = boundary_iou(gt_numpy, pred_numpy, dilation_ratio=0.003)
            all_biou_scores.append(biou_score)

        except Exception as e:
            print(f"\nError processing image {os.path.basename(mask_path)}: {e}. Skipped this file.")
            continue

    if all_iou_scores:
        average_iou = np.mean(all_iou_scores)
        average_f1 = np.mean(all_f1_scores)
        average_bf1 = np.mean(all_bf1_scores)
        average_bssim = np.nanmean(all_bssim_scores)
        average_biou = np.mean(all_biou_scores)

        print(f"\n{'=' * 70}")
        print(f"{'Final Results Summary':^62}")
        print(f"{'=' * 70}")
        print(f"  Total successfully processed images: {len(all_iou_scores)}")
        print(f"{'-' * 70}")
        print(f"  {'Metric Name':<25} | {'Average':<10}")
        print(f"{'-' * 70}")
        print(f"  {'IoU':<25} | {average_iou:.4f}")
        print(f"  {'F1':<25} | {average_f1:.4f}")
        print(f"  {'B-F1':<25} | {average_bf1:.4f}")
        print(f"  {'B-SSIM':<25} | {average_bssim:.4f}")
        print(f"  {'B-IoU':<25} | {average_biou:.4f}")
        print(f"{'=' * 70}")
    else:
        print("No images were successfully processed, cannot calculate metrics.")

if __name__ == "__main__":
    main()
