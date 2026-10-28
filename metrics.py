"""Dense pixel- and boundary-level evaluation metrics.

Shared by ``test.py`` (model evaluation) and ``test_vae.py`` (VAE
reconstruction evaluation). Boundary-level definitions follow the paper:
B-F1 compares one-pixel inner boundaries with a Euclidean distance tolerance
of two pixels; B-SSIM uses an 11x11 uniformly weighted window with
``data_range=1`` averaged on the one-pixel ground-truth inner boundary; and
B-IoU follows Cheng et al. with two-pixel boundary bands at 512x512 tiles.
Polygon-level geometry metrics (C-IoU, Vertex F1, PoLiS) live in
``evaluate_geometry.py``.
"""

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


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
