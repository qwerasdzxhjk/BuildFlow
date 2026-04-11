import os
# Set CUDA device before any PyTorch imports
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from pathlib import Path
import ml_collections
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from absl import app, flags
from absl import logging as absl_logging
from ml_collections import config_flags
from tqdm.auto import tqdm
import einops
from peft import PeftModel
import collections

# Assume utils, libs, and diffusion modules are in the Python search path
import utils
import libs.autoencoder
from diffusion.flow_matching import ODEEulerFlowMatchingSolver

# --- Command Line Flags ---
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Configuration file.", lock_config=True)
flags.mark_flags_as_required(["config"])

# ==================== Metric Calculation Functions ====================

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

# ==================== Dataset & Pipeline ====================

class InferenceImageDataset(Dataset):
    def __init__(self, folder_path, image_size, label_folder_path=None, exts=None):
        super().__init__()
        if exts is None:
            exts = ['tif', 'jpg', 'jpeg', 'png', 'webp', 'tiff']
        self.image_size = image_size
        self.paths = sorted([p for ext in exts for p in Path(folder_path).glob(f'**/*.{ext}')])
        self.label_folder_path = Path(label_folder_path) if label_folder_path else None
        
        if not self.paths:
            raise FileNotFoundError(f"No images with extensions {exts} found in {folder_path}")
        print(f"Found {len(self.paths)} images in {folder_path}")
        
        if self.label_folder_path:
            print(f"Will look for corresponding labels in: {self.label_folder_path}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        item = {'filename': path.stem}
        try:
            pil_image = Image.open(path).convert('RGB')
            image_arr = np.array(pil_image)
            image_tensor = (image_arr / 127.5 - 1.0).astype(np.float32)
            image_tensor = torch.from_numpy(einops.rearrange(image_tensor, 'h w c -> c h w'))
            item['image'] = image_tensor
            
            if self.label_folder_path:
                label_path = None
                for ext in ['png', 'tif', 'jpg', 'jpeg']:
                    potential_path = self.label_folder_path / f"{path.stem}.{ext}"
                    if potential_path.exists():
                        label_path = potential_path
                        break
                if label_path:
                    label_pil = Image.open(label_path)
                    label_arr = np.array(label_pil)
                    label_tensor = (label_arr > 0).astype(np.int64)
                    item['label'] = torch.from_numpy(label_tensor)
                else:
                    print(f"Warning: Label for {path.name} not found. Metrics will not be calculated for this image.")
            return item
        except Exception as e:
            print(f"Warning: Could not load or process image {path}. Skipping. Error: {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_test_config(config: ml_collections.ConfigDict):
    test_config = config.get('test')
    if test_config is None:
        raise ValueError("config.test is missing. Please define test settings in your config file before running test.py.")

    if not test_config.get('ckpt_dir'):
        raise ValueError("config.test.ckpt_dir is empty. Please set it in your config file before running test.py.")
    if not test_config.get('input_dir'):
        raise ValueError("config.test.input_dir is empty. Please set it in your config file before running test.py.")
    if not test_config.get('output_dir'):
        raise ValueError("config.test.output_dir is empty. Please set it in your config file before running test.py.")

    return test_config

def sample(config: ml_collections.ConfigDict):
    test_config = get_test_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    
    os.makedirs(test_config.output_dir, exist_ok=True)
    utils.set_logger(log_level='info', fname=os.path.join(test_config.output_dir, 'output.log'))
    
    absl_logging.info(f"Using checkpoint directory: {test_config.ckpt_dir}")
    if test_config.label_dir:
        absl_logging.info(f"Calculating metrics using labels from: {test_config.label_dir}")

    mask_output_dir = None
    if test_config.save_pred_masks:
        mask_output_dir = test_config.pred_mask_dir if test_config.pred_mask_dir else os.path.join(test_config.output_dir, 'pred_masks')
        absl_logging.info(f"Saving predicted masks to: {mask_output_dir}")
        os.makedirs(mask_output_dir, exist_ok=True)

    absl_logging.info(f'Process using device: {device}')

    dataset = InferenceImageDataset(test_config.input_dir, config.dataset.image_size, label_folder_path=test_config.label_dir)
    if test_config.n_samples > 0:
        dataset.paths = dataset.paths[:test_config.n_samples]
    dataloader = DataLoader(
        dataset,
        batch_size=test_config.batch_size,
        shuffle=False,
        num_workers=test_config.get('num_workers', 4),
        collate_fn=collate_fn
    )

    absl_logging.info("Initializing base model structure with PEFT config...")
    train_state = utils.initialize_train_state(config, device, layers_to_fully_train=['flow_model.x_embedder'])
    base_nnet_ema = train_state.nnet_ema
    
    if test_config.base_model_path and os.path.exists(test_config.base_model_path):
        absl_logging.info(f"Attempting to load and remap pre-trained weights from: {test_config.base_model_path}")
        try:
            pre_weights = torch.load(test_config.base_model_path, map_location='cpu')
            if 'state_dict' in pre_weights: pre_weights = pre_weights['state_dict']
            elif 'model' in pre_weights: pre_weights = pre_weights['model']
            elif 'nnet' in pre_weights: pre_weights = pre_weights['nnet']
            
            new_state_dict = collections.OrderedDict()
            shared_prefixes = ['x_embedder.', 't_embedder.', 'y_embedder.', 'blocks.', 'final_layer.']
            transferred_count = 0
            for key, value in pre_weights.items():
                is_shared = False
                if key == 'pos_embed':
                    is_shared = True
                else:
                    for prefix in shared_prefixes:
                        if key.startswith(prefix):
                            is_shared = True
                            break
                if is_shared:
                    new_key = f'flow_model.{key}'
                    new_state_dict[new_key] = value
                    transferred_count += 1
                    
            absl_logging.info(f"Remapped and filtered weights. Transferred {transferred_count} tensors.")
            target_model = base_nnet_ema.base_model.model
            missing_keys, unexpected_keys = target_model.load_state_dict(new_state_dict, strict=False)
            absl_logging.info("Pre-trained base model weights loaded successfully.")
        except Exception as e:
            absl_logging.error(f"Failed to load and remap pre-trained weights: {e}", exc_info=True)
            raise e

    lora_adapter_path = os.path.join(test_config.ckpt_dir, 'lora_ema')
    if not os.path.isdir(lora_adapter_path):
        raise FileNotFoundError(f"LoRA EMA adapter directory not found: {lora_adapter_path}")
    absl_logging.info(f'Loading LoRA EMA adapter from {lora_adapter_path}')
    base_nnet_ema.load_adapter(lora_adapter_path, adapter_name="default")
    
    training_state_path = os.path.join(test_config.ckpt_dir, 'training_state.pt')
    if not os.path.exists(training_state_path):
        raise FileNotFoundError(f"Required 'training_state.pt' not found: {test_config.ckpt_dir}")
    state = torch.load(training_state_path, map_location=device)
    if 'x_embedder' in state:
        base_nnet_ema.flow_model.x_embedder.modules_to_save['default'].load_state_dict(state['x_embedder'])
    else:
        raise ValueError("'x_embedder' weights not found in training_state.pt.")

    nnet_ema = base_nnet_ema
    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    nnet_ema.to(device)
    autoencoder.to(device)
    nnet_ema.eval()
    autoencoder.eval()
    absl_logging.info("Models prepared for inference.")

    @torch.no_grad()
    def ode_fm_solver_sample(model, source_data, n_samples, sample_steps, cfg_scale):
        @torch.cuda.amp.autocast()
        def encode(_batch): return autoencoder.encode(_batch)
        @torch.cuda.amp.autocast()
        def decode(_batch): return autoencoder.decode(_batch)
        z_0 = encode(source_data)
        ode_solver = ODEEulerFlowMatchingSolver(model, source_latent=z_0, has_null_indicator=True)
        _z, _ = ode_solver.sample(
            x_T=z_0, batch_size=n_samples, sample_steps=sample_steps,
            unconditional_guidance_scale=cfg_scale, has_null_indicator=1,
        )
        return decode(_z)

    if test_config.get('sample_steps_list'):
        steps_to_iterate = test_config.sample_steps_list
        absl_logging.info(f"Iterating over sampling steps: {steps_to_iterate}")
    else:
        single_step = test_config.get('sample_steps', config.sample.sample_steps)
        steps_to_iterate = [single_step]
        absl_logging.info(f"Using single sampling step: {single_step}")

    cfg_scale = test_config.get('cfg_scale', config.sample.scale)
    threshold = test_config.get('threshold', 0.5)
    absl_logging.info(f"Using CFG scale: {cfg_scale}, Threshold: {threshold}")

    def unpreprocess(x):
        x = 0.5 * (x + 1.0)
        x.clamp_(0.0, 1.0)
        return x

    iou_scores = {s: [] for s in steps_to_iterate}
    f1_scores = {s: [] for s in steps_to_iterate}
    bf1_scores = {s: [] for s in steps_to_iterate}
    bssim_scores = {s: [] for s in steps_to_iterate}
    biou_scores = {s: [] for s in steps_to_iterate}

    pbar = tqdm(total=len(dataset), desc="Generating Images & Calculating Metrics")
    
    for batch in dataloader:
        if batch is None: continue
        source_images = batch['image'].to(device)
        filenames = batch['filename']
        labels = batch.get('label')
        if labels is not None:
            labels = labels.to(device)
            
        for current_steps in steps_to_iterate:
            generated_images = ode_fm_solver_sample(
                nnet_ema, source_images, source_images.shape[0], current_steps, cfg_scale
            )

            generated_images_cpu = generated_images.cpu()
            if labels is not None:
                labels_cpu = labels.cpu()
                
            generated_unpreprocessed = unpreprocess(generated_images_cpu)
            generated_gray = generated_unpreprocessed.mean(dim=1)
            predicted_masks = (generated_gray > threshold)
            
            if test_config.save_pred_masks and mask_output_dir:
                for idx, fname in enumerate(filenames):
                    mask_save_path = os.path.join(mask_output_dir, f"{fname}.png")
                    mask_uint8 = (predicted_masks[idx].byte().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(mask_uint8).save(mask_save_path)

            for i in range(len(filenames)):
                if labels is not None:
                    pred_mask_i = predicted_masks[i].float()
                    label_i = labels_cpu[i].float()
                    
                    pred_numpy = predicted_masks[i].byte().cpu().numpy()
                    gt_numpy = labels_cpu[i].byte().cpu().numpy()

                    current_iou = calculate_iou(pred_mask_i, label_i)
                    iou_scores[current_steps].append(current_iou)

                    current_f1 = calculate_f1_score(pred_mask_i, label_i)
                    f1_scores[current_steps].append(current_f1)

                    current_bf1 = calculate_bf1_matlab(pred_numpy, gt_numpy, distance_threshold=2.0)
                    bf1_scores[current_steps].append(current_bf1)

                    current_bssim = calculate_bssim_strict(pred_numpy, gt_numpy)
                    bssim_scores[current_steps].append(current_bssim)

                    current_biou = boundary_iou(gt_numpy, pred_numpy, dilation_ratio=0.003)
                    biou_scores[current_steps].append(current_biou)

        pbar.update(len(filenames))
    pbar.close()

    absl_logging.info("=" * 70)
    absl_logging.info(f"{'Final Results Summary':^62}")
    absl_logging.info("=" * 70)
    
    if any(iou_scores.values()):
        for steps, scores in iou_scores.items():
            if scores:
                absl_logging.info(f"--- Steps: {steps} ---")
                absl_logging.info(f"  Total successfully processed images: {len(scores)}")
                absl_logging.info("-" * 70)
                absl_logging.info(f"  {'Metric Name':<25} | {'Average':<10}")
                absl_logging.info("-" * 70)
                
                mean_iou = np.mean(scores)
                absl_logging.info(f"  {'IoU':<25} | {mean_iou:.4f}")
                
                mean_f1 = np.mean(f1_scores[steps])
                absl_logging.info(f"  {'F1':<25} | {mean_f1:.4f}")
                
                mean_bf1 = np.mean(bf1_scores[steps])
                absl_logging.info(f"  {'B-F1':<25} | {mean_bf1:.4f}")
                
                mean_bssim = np.nanmean(bssim_scores[steps]) 
                absl_logging.info(f"  {'B-SSIM':<25} | {mean_bssim:.4f}")

                mean_biou = np.mean(biou_scores[steps])
                absl_logging.info(f"  {'B-IoU':<25} | {mean_biou:.4f}")
                
    absl_logging.info("=" * 70)

def main(argv):
    config = FLAGS.config
    sample(config)

if __name__ == "__main__":
    app.run(main)
