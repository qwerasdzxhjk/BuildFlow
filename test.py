"""BuildFlow inference and evaluation on a test/validation split.

Loads the frozen CrossFlow base model plus the trained LoRA adapter and
x_embedder, runs the deterministic Euler ODE integration from the encoded
input image to the predicted mask, and reports dense pixel- and boundary-level
metrics (IoU, F1, B-F1, B-SSIM, B-IoU). Predicted masks can be saved for the
polygon-level geometry evaluation (evaluate_geometry.py).
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get(
    'BUILD_FLOW_CUDA_VISIBLE_DEVICES',
    os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
)
from pathlib import Path
import ml_collections
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from absl import app, flags
from absl import logging as absl_logging
from ml_collections import config_flags
from tqdm.auto import tqdm
import einops
import collections

# Assume utils, libs, and diffusion modules are in the Python search path
import utils
import libs.autoencoder
from diffusion.flow_matching import ODEEulerFlowMatchingSolver

from metrics import (
    boundary_iou,
    calculate_bf1_matlab,
    calculate_bssim_strict,
    calculate_f1_score,
    calculate_iou,
)

# --- Command Line Flags ---
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Configuration file.", lock_config=True)
flags.mark_flags_as_required(["config"])

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
            try:
                pre_weights = torch.load(
                    test_config.base_model_path, map_location='cpu', weights_only=False
                )
            except TypeError:  # older torch without the weights_only kwarg
                pre_weights = torch.load(test_config.base_model_path, map_location='cpu')
            if 'state_dict' in pre_weights: pre_weights = pre_weights['state_dict']
            elif 'model' in pre_weights: pre_weights = pre_weights['model']
            elif 'nnet' in pre_weights: pre_weights = pre_weights['nnet']
            
            new_state_dict = collections.OrderedDict()
            shared_prefixes = ['x_embedder.', 't_embedder.', 'y_embedder.', 'blocks.', 'final_layer.']
            transferred_count = 0
            pretrained_x_weight = None
            pretrained_x_bias = None
            for key, value in pre_weights.items():
                # The PEFT wrapper stores x_embedder under
                # ``modules_to_save.default`` and the I2I model has twice the
                # input channels.  Handle it explicitly below rather than
                # silently ignoring it under strict=False.
                if key == 'x_embedder.proj.weight':
                    pretrained_x_weight = value
                    continue
                if key == 'x_embedder.proj.bias':
                    pretrained_x_bias = value
                    continue
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
            if pretrained_x_weight is not None:
                embedder = base_nnet_ema.flow_model.x_embedder.modules_to_save['default']
                weight = embedder.proj.weight.data.clone()
                original_channels = pretrained_x_weight.shape[1]
                if original_channels > weight.shape[1]:
                    raise ValueError(
                        f"Pretrained x_embedder has {original_channels} input channels, "
                        f"but the I2I model accepts {weight.shape[1]}."
                    )
                weight[:, :original_channels].copy_(pretrained_x_weight)
                for channel in range(original_channels, weight.shape[1]):
                    weight[:, channel].copy_(pretrained_x_weight[:, channel % original_channels])
                embedder.proj.weight.data.copy_(weight)
                if pretrained_x_bias is not None:
                    embedder.proj.bias.data.copy_(pretrained_x_bias)
            if missing_keys or unexpected_keys:
                absl_logging.warning(
                    "Base checkpoint key differences (missing=%s, unexpected=%s)",
                    missing_keys[:8], unexpected_keys[:8],
                )
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
    try:
        state = torch.load(training_state_path, map_location=device, weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        state = torch.load(training_state_path, map_location=device)
    # The EMA network must use the EMA copy saved alongside the EMA LoRA
    # adapter.  Fall back to the main weights for backwards compatibility.
    embedder_state = state.get('x_embedder_ema', state.get('x_embedder'))
    if embedder_state is not None:
        base_nnet_ema.flow_model.x_embedder.modules_to_save['default'].load_state_dict(embedder_state)
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
