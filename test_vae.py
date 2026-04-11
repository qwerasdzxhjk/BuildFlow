import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import argparse
from PIL import Image
import numpy as np
import glob
import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import libs.autoencoder

def to_binary_numpy(tensor, threshold=0.5):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach().numpy()
    if tensor.max() > 1:
        return (tensor > 127).astype(np.uint8)
    else:
        return (tensor > threshold).astype(np.uint8)

def to_binary_tensor(tensor, threshold=0.5):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.max() > 1:
        return (tensor > 127).float()
    else:
        return (tensor > threshold).float()

def mask_to_boundary_erode(mask, dilation_ratio=0.01):
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    return mask - mask_erode

def boundary_iou(gt, dt, dilation_ratio=0.003):
    gt = to_binary_numpy(gt)
    dt = to_binary_numpy(dt)
    
    gt_boundary = mask_to_boundary_erode(gt, dilation_ratio)
    dt_boundary = mask_to_boundary_erode(dt, dilation_ratio)
    
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    
    if union == 0:
        return 1.0
    
    return intersection / union

def get_boundary_strict(mask_uint8):
    pad_mask = cv2.copyMakeBorder(mask_uint8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_pad = cv2.erode(pad_mask, kernel)
    eroded = eroded_pad[1:-1, 1:-1]
    boundary = mask_uint8 - eroded
    return boundary

def calculate_bf1_matlab(pred_bin, gt_bin, distance_threshold=2.0):
    pred_bin = to_binary_numpy(pred_bin)
    gt_bin = to_binary_numpy(gt_bin)

    pred_boundary = get_boundary_strict(pred_bin)
    gt_boundary = get_boundary_strict(gt_bin)

    pred_b_sum = np.sum(pred_boundary)
    gt_b_sum = np.sum(gt_boundary)

    if gt_b_sum == 0 and pred_b_sum == 0:
        return 1.0
    if gt_b_sum == 0 or pred_b_sum == 0:
        return 0.0

    gt_dist_map = cv2.distanceTransform(1 - gt_boundary, cv2.DIST_L2, 5)
    pred_dists = gt_dist_map[pred_boundary > 0]
    precision = np.sum(pred_dists <= distance_threshold) / (pred_b_sum + 1e-6)

    pred_dist_map = cv2.distanceTransform(1 - pred_boundary, cv2.DIST_L2, 5)
    gt_dists = pred_dist_map[gt_boundary > 0]
    recall = np.sum(gt_dists <= distance_threshold) / (gt_b_sum + 1e-6)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1

def calculate_bssim_strict(pred_bin, gt_bin):
    pred_bin = to_binary_numpy(pred_bin).astype(np.float32)
    gt_bin_f = to_binary_numpy(gt_bin).astype(np.float32)
    gt_boundary = get_boundary_strict(to_binary_numpy(gt_bin)).astype(np.float32)

    if np.sum(gt_boundary) == 0:
        return np.nan

    H, W = pred_bin.shape
    win_size = min(H, W, 11)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return np.nan

    _, ssim_map = ssim(gt_bin_f, pred_bin, win_size=win_size, data_range=1.0, full=True)
    return (ssim_map * gt_boundary).sum() / (gt_boundary.sum() + 1e-6)

def calculate_iou(pred, label, eps=1e-6):
    pred = to_binary_tensor(pred)
    label = to_binary_tensor(label)
    intersection = (pred * label).sum()
    union = pred.sum() + label.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / (union + eps)).item()

def calculate_f1_score(pred, label):
    pred = to_binary_tensor(pred)
    label = to_binary_tensor(label)
    tp = (pred * label).sum()
    denominator = pred.sum() + label.sum()
    if denominator == 0:
        return 1.0
    return (2 * tp / (denominator + 1e-6)).item()

def load_ground_truth_mask(image_path: str):
    mask = Image.open(image_path).convert("L")
    mask_np = np.array(mask)
    mask_np = (mask_np > 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_np)
    return mask_tensor

def preprocess_image(image_path: str, height: int, width: int):
    image = Image.open(image_path).convert("L")
    image_tensor = np.array(image)
    image_tensor = image_tensor.astype(np.float32) * 255.0
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0)
    image_tensor = image_tensor.repeat(3, 1, 1)
    image_tensor = (image_tensor / 255.0 - 0.5) * 2.0
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
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
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
            input_tensor = preprocess_image(mask_path, args.height, args.width)

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
