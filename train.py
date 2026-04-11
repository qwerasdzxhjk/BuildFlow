import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from pathlib import Path
import tempfile
import collections
import numpy as np
import shutil

import ml_collections
import torch
from torch import multiprocessing as mp
from torch.cuda.amp import GradScaler
from torchvision.utils import make_grid, save_image
from absl import app, flags
from absl import logging as absl_logging
from ml_collections import config_flags
from tqdm.auto import tqdm
import builtins
import glob
from peft import LoraConfig, get_peft_model, PeftModel

import utils
import libs.autoencoder
from datasets import create_i2i_dataloaders
from diffusion.flow_matching_loss import ModalityFlowMatching
from diffusion.flow_matching import ODEEulerFlowMatchingSolver

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")

def train(config: ml_collections.ConfigDict):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    config.mixed_precision = 'fp16' if torch.cuda.is_available() else 'no'
    config = ml_collections.FrozenConfigDict(config)

    os.makedirs(config.ckpt_root, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
    absl_logging.info(config)

    absl_logging.info(f'Process using device: {device}')

    train_loader, val_loader, data_factory = create_i2i_dataloaders(config)
    absl_logging.info("Data loaders created.")

    train_state = utils.initialize_train_state(config, device, layers_to_fully_train=['flow_model.x_embedder'])
    
    ckpt_dirs = sorted(glob.glob(os.path.join(config.ckpt_root, '[0-9]'*7)))
    if len(ckpt_dirs) > 0:
        latest_ckpt_dir = ckpt_dirs[-1]
        absl_logging.info(f"Found latest checkpoint: {latest_ckpt_dir}")
        state_path = os.path.join(latest_ckpt_dir, 'training_state.pt')
        if os.path.exists(state_path):
            absl_logging.info(f"Resuming training state from {state_path}")
            state = torch.load(state_path, map_location=device)
            train_state.step = state['step']
            train_state.optimizer.load_state_dict(state['optimizer'])
            train_state.lr_scheduler.load_state_dict(state['lr_scheduler'])

            if 'x_embedder' in state:
                absl_logging.info("Loading fully fine-tuned x_embedder weights from checkpoint.")
                train_state.nnet.flow_model.x_embedder.modules_to_save['default'].load_state_dict(state['x_embedder'])
            else:
                absl_logging.warning("Checkpoint does not contain 'x_embedder' weights. It might be an old checkpoint.")
            
            lora_main_path = os.path.join(latest_ckpt_dir, 'lora_main', 'adapter_model.bin')
            lora_ema_path = os.path.join(latest_ckpt_dir, 'lora_ema', 'adapter_model.bin')
            if os.path.exists(lora_main_path):
                 absl_logging.info(f"Loading LoRA main adapter from: {lora_main_path}")
                 main_adapter_weights = torch.load(lora_main_path, map_location=device)
                 train_state.nnet.load_state_dict(main_adapter_weights, strict=False)
            if os.path.exists(lora_ema_path):
                 absl_logging.info(f"Loading LoRA ema adapter from: {lora_ema_path}")
                 ema_adapter_weights = torch.load(lora_ema_path, map_location=device)
                 train_state.nnet_ema.load_state_dict(ema_adapter_weights, strict=False)
        else:
            absl_logging.warning(f"Checkpoint directory {latest_ckpt_dir} found, but training_state.pt is missing. Starting from scratch.")
    else:
        absl_logging.info("No checkpoint found, starting from scratch.")


    pretrained_ckpt_path = config.train.get('pretrained_ckpt_path')
    if pretrained_ckpt_path and train_state.step == 0:
        absl_logging.info(f"Attempting to load pre-trained weights from: {pretrained_ckpt_path}")
        if os.path.exists(pretrained_ckpt_path):
            try:
                pre_weights = torch.load(pretrained_ckpt_path, map_location='cpu')
                if 'state_dict' in pre_weights: pre_weights = pre_weights['state_dict']
                elif 'model' in pre_weights: pre_weights = pre_weights['model']
                elif 'nnet' in pre_weights: pre_weights = pre_weights['nnet']
                
                new_state_dict = collections.OrderedDict()
                shared_prefixes = ['x_embedder.', 't_embedder.', 'y_embedder.', 'blocks.', 'final_layer.']
                transferred_count = 0
                original_x_embedder_weight = None
                original_x_embedder_bias = None

                for key, value in pre_weights.items():
                    if key == 'x_embedder.proj.weight':
                        original_x_embedder_weight = value
                        continue 
                    if key == 'x_embedder.proj.bias':
                        original_x_embedder_bias = value
                        continue 

                    is_shared = False
                    if key == 'pos_embed': is_shared = True
                    else:
                        for prefix in shared_prefixes:
                            if key.startswith(prefix):
                                is_shared = True
                                break
                    if is_shared:
                        new_key = f'flow_model.{key}'
                        new_state_dict[new_key] = value
                        transferred_count += 1
                
                if original_x_embedder_weight is not None:
                    absl_logging.info("Found original x_embedder weights. Proceeding with adaptation for DVI.")
                    current_embedder = train_state.nnet.flow_model.x_embedder.modules_to_save['default']
                    new_weight = current_embedder.proj.weight.data.clone()
                    
                    in_channels_orig = original_x_embedder_weight.shape[1]
                    new_weight[:, :in_channels_orig, :, :] = original_x_embedder_weight
                    
                    init_strategy = config.train.get('dvi_init_strategy', 'zero')
                    if init_strategy == 'random':
                        torch.nn.init.kaiming_normal_(new_weight[:, in_channels_orig:, :, :])
                        absl_logging.info("DVI Ablation: Initialized new channels with Random (Kaiming Normal).")
                    elif init_strategy == 'copy':
                        in_total = new_weight.shape[1]
                        for j in range(in_channels_orig, in_total):
                            new_weight[:, j, :, :] = original_x_embedder_weight[:, j % in_channels_orig, :, :]
                        absl_logging.info("DVI Ablation: Initialized new channels by Copying original weights.")
                    else:
                        new_weight[:, in_channels_orig:, :, :].zero_()
                        absl_logging.info("DVI Ablation: Initialized new channels with Zeros.")
                    
                    new_state_dict['flow_model.x_embedder.modules_to_save.default.proj.weight'] = new_weight
                    if original_x_embedder_bias is not None:
                        new_state_dict['flow_model.x_embedder.modules_to_save.default.proj.bias'] = original_x_embedder_bias
                    absl_logging.info("Adapted x_embedder weights for Direct Video Injection (DVI).")
                    transferred_count += 1 
                
                absl_logging.info(f"Remapped and filtered weights. Transferred {transferred_count} tensors.")
                
                train_state.nnet.base_model.model.load_state_dict(new_state_dict, strict=False)
                absl_logging.info("Pre-trained weights loaded successfully into the new model structure.")
                
                train_state.nnet_ema.load_state_dict(train_state.nnet.state_dict())
                absl_logging.info("EMA model weights synchronized with the loaded pre-trained weights.")
            except Exception as e:
                absl_logging.error(f"Failed to load and remap pre-trained weights: {e}", exc_info=True)
        else:
            absl_logging.warning(f"Pre-trained checkpoint not found at: {pretrained_ckpt_path}")
    elif pretrained_ckpt_path and train_state.step > 0:
         absl_logging.info(f"Skipping loading pre-trained weights because training is being resumed from step {train_state.step}.")

    nnet = train_state.nnet
    nnet_ema = train_state.nnet_ema
    optimizer = train_state.optimizer
    lr_scheduler = train_state.lr_scheduler
    absl_logging.info("Model and optimizer prepared for LoRA fine-tuning.")

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)
    autoencoder.eval()

    flow_matcher = ModalityFlowMatching()
    absl_logging.info("VAE and Flow Matcher initialized.")

    scaler = GradScaler()

    def encode(_batch):
        with torch.no_grad():
            return autoencoder.encode(_batch)

    def decode(_batch):
        with torch.no_grad():
            return autoencoder.decode(_batch)

    def get_data_generator(loader):
        while True:
            for data in loader:
                yield data

    data_generator = get_data_generator(train_loader)
    val_generator = get_data_generator(val_loader)

    def train_step(_batch):
        optimizer.zero_grad()
        
        if config.get('use_preprocessed', False):
            source_latent = autoencoder.sample(_batch['source_latent'])
            target_latent = autoencoder.sample(_batch['target_latent'])
        else:
            source_latent = encode(_batch['source_images'])
            target_latent = encode(_batch['target_images'])
            
        with torch.cuda.amp.autocast():
            loss, loss_dict = flow_matcher(
                nnet=nnet,
                source_latent=source_latent,
                target_latent=target_latent,
                target_mask=_batch['target_masks'],
                autoencoder=autoencoder,
                all_config=config,
            )
        
        scaler.scale(loss).backward()
        if config.train.get('grad_clip', -1) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(nnet.parameters(), config.train.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        train_state.ema_update(config.train.ema_rate)
        train_state.step += 1

        metrics = {'loss': loss.detach().item()}
        for key, val in loss_dict.items():
            metrics[key] = val.detach().mean().item()
        metrics['lr'] = optimizer.param_groups[0]['lr']

        return metrics

    @torch.no_grad()
    def ode_fm_solver_sample(model, source_data, n_samples, sample_steps):
        z_0 = encode(source_data)
        ode_solver = ODEEulerFlowMatchingSolver(
            model,
            source_latent=z_0,
            has_null_indicator=True
        )
        _z, _ = ode_solver.sample(
            x_T=z_0,
            batch_size=n_samples,
            sample_steps=sample_steps,
            unconditional_guidance_scale=config.sample.scale,
            has_null_indicator=1,
        )
        return decode(_z)

    def log_sample_images(sample_config):
        absl_logging.info(f'Generating a sample grid for qualitative evaluation (in serial mode)...')
        model_for_eval = nnet_ema
        model_for_eval.eval()

        val_batch = next(val_generator)
        for k, v in val_batch.items():
            if isinstance(v, torch.Tensor):
                val_batch[k] = v.to(device)
        
        n_samples = min(sample_config.n_samples, val_batch['source_images'].size(0))
        
        source_images_batch = val_batch['source_images'][:n_samples].to(device)
        target_images_batch = val_batch['target_images'][:n_samples].to(device)

        generated_samples = []
        
        absl_logging.info(f"Starting serial sampling for {n_samples} images...")
        for i in tqdm(range(n_samples), desc="Serial Sampling"):
            single_source_image = source_images_batch[i].unsqueeze(0)
            
            single_sample = ode_fm_solver_sample(
                model_for_eval, 
                single_source_image, 
                n_samples=1,
                sample_steps=sample_config.sample_steps
            )
            
            generated_samples.append(single_sample.squeeze(0).cpu())

        absl_logging.info("Serial sampling finished. Combining images into a grid.")
        
        samples = torch.stack(generated_samples).to(device)

        source_unpreprocessed = data_factory.unpreprocess(source_images_batch)
        samples_unpreprocessed = data_factory.unpreprocess(samples)
        target_unpreprocessed = data_factory.unpreprocess(target_images_batch)
        target_unpreprocessed = target_unpreprocessed * 255
        comparison_grid = torch.cat([source_unpreprocessed, samples_unpreprocessed, target_unpreprocessed])
        grid = make_grid(comparison_grid, nrow=n_samples)
        save_path = os.path.join(config.sample_dir, f'sample_{train_state.step:07d}.png')
        save_image(grid, save_path)
        absl_logging.info(f'Saved sample grid to {save_path}')

        model_for_eval.train()

    def calculate_iou(pred, label, smooth=1e-6):
        pred = pred.long()
        label = label.long()
        intersection = (pred & label).sum()
        union = (pred | label).sum()
        iou = (intersection.float() + smooth) / (union.float() + smooth)
        return iou.item()

    @torch.no_grad()
    def eval_step(sample_config):
        absl_logging.info(f'Starting quantitative evaluation on validation set...')
        model_for_eval = nnet_ema
        model_for_eval.eval()

        all_iou_scores = []
        threshold = sample_config.get('threshold', 0.5)
        absl_logging.info(f'Using threshold {threshold} for binarization.')

        for batch in tqdm(val_loader, desc="Validation"):
            source_images = batch['source_images'].to(device)
            target_masks = batch['target_masks'].to(device)

            generated_images = ode_fm_solver_sample(
                model_for_eval,
                source_images,
                n_samples=source_images.shape[0],
                sample_steps=sample_config.sample_steps
            )

            generated_unpreprocessed = data_factory.unpreprocess(generated_images)
            generated_gray = generated_unpreprocessed.mean(dim=1)
            predicted_masks = (generated_gray > threshold).squeeze(1)

            for i in range(predicted_masks.shape[0]):
                true_mask = target_masks[i].squeeze()
                pred_mask = predicted_masks[i]
                iou = calculate_iou(pred_mask, true_mask)
                all_iou_scores.append(iou)

        mean_iou = 0.0
        if not all_iou_scores:
            absl_logging.warning("No IoU scores were calculated. The validation set might be empty.")
        else:
            mean_iou = np.mean(all_iou_scores)
            absl_logging.info(f'Validation complete. Mean Foreground IoU: {mean_iou:.4f}')

        model_for_eval.train()
        return mean_iou

    absl_logging.info(f'Start training from step {train_state.step}...')
    pbar = tqdm(initial=train_state.step, total=config.train.n_steps)
    best_iou_so_far = 0.0

    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = next(data_generator)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        metrics = train_step(batch)
        pbar.update(1)

        if train_state.step % config.train.log_interval == 0:
            desc = f"Step: {train_state.step}"
            if 'loss' in metrics: desc += f" | Loss: {metrics['loss']:.4f}"
            if 'loss_diff' in metrics: desc += f" | Diff: {metrics['loss_diff']:.4f}"
            if 'loss_recon_dice' in metrics: desc += f" | Seg: {metrics['loss_recon_dice']:.4f}"
            if 'lr' in metrics: desc += f" | LR: {metrics['lr']:.6f}"
            pbar.set_description(desc)
            log_message = f"Step: {train_state.step}"
            for key, val in metrics.items():
                log_message += f" | {key}: {val:.6f}"
            absl_logging.info(log_message)

        start_eval_step = config.train.get('start_eval_step', 0)
        if train_state.step >= start_eval_step and train_state.step % config.train.save_interval == 0:
            torch.cuda.empty_cache()
            eval_iou = eval_step(config.sample)
            torch.cuda.empty_cache()

            periodic_save_dir = os.path.join(config.ckpt_root, f'{train_state.step:07d}')
            os.makedirs(periodic_save_dir, exist_ok=True)
            train_state.nnet.save_pretrained(os.path.join(periodic_save_dir, 'lora_main'))
            train_state.nnet_ema.save_pretrained(os.path.join(periodic_save_dir, 'lora_ema'))
            
            periodic_training_state = {
                'step': train_state.step,
                'optimizer': train_state.optimizer.state_dict(),
                'lr_scheduler': train_state.lr_scheduler.state_dict(),
                'x_embedder': train_state.nnet.flow_model.x_embedder.modules_to_save['default'].state_dict()
            }
            torch.save(periodic_training_state, os.path.join(periodic_save_dir, 'training_state.pt'))
            absl_logging.info(f"Saved periodic checkpoint to {periodic_save_dir}")

            if config.train.get('max_checkpoints_to_keep', -1) > 0:
                all_ckpts = sorted(glob.glob(os.path.join(config.ckpt_root, '[0-9]'*7)))
                if len(all_ckpts) > config.train.max_checkpoints_to_keep:
                    ckpts_to_delete = all_ckpts[:-config.train.max_checkpoints_to_keep]
                    for ckpt in ckpts_to_delete:
                        absl_logging.info(f"Removing old checkpoint {ckpt}")
                        shutil.rmtree(ckpt)

            if eval_iou > best_iou_so_far:
                absl_logging.info(f"New best model found! IoU improved from {best_iou_so_far:.4f} to {eval_iou:.4f}. Saving checkpoint.")
                best_iou_so_far = eval_iou
                
                best_save_path = os.path.join(config.ckpt_root, 'best_model')
                os.makedirs(best_save_path, exist_ok=True)
                
                lora_main_save_path = os.path.join(best_save_path, 'lora_main')
                lora_ema_save_path = os.path.join(best_save_path, 'lora_ema')
                train_state.nnet.save_pretrained(lora_main_save_path)
                train_state.nnet_ema.save_pretrained(lora_ema_save_path)
                
                training_state = {
                    'step': train_state.step,
                    'optimizer': train_state.optimizer.state_dict(),
                    'lr_scheduler': train_state.lr_scheduler.state_dict(),
                    'x_embedder': train_state.nnet.flow_model.x_embedder.modules_to_save['default'].state_dict()
                }
                torch.save(training_state, os.path.join(best_save_path, 'training_state.pt'))
                
                absl_logging.info(f'Saved best LoRA checkpoint to {best_save_path}')
            else:
                absl_logging.info(f"Model did not improve. Current IoU: {eval_iou:.4f}, Best IoU: {best_iou_so_far:.4f}. Skipping best model save.")

    pbar.close()
    final_save_path = os.path.join(config.ckpt_root, 'final')
    os.makedirs(final_save_path, exist_ok=True)
    train_state.nnet.save_pretrained(os.path.join(final_save_path, 'lora_main'))
    train_state.nnet_ema.save_pretrained(os.path.join(final_save_path, 'lora_ema'))
    final_training_state = {
        'step': train_state.step,
        'optimizer': train_state.optimizer.state_dict(),
        'lr_scheduler': train_state.lr_scheduler.state_dict(),
        'x_embedder': train_state.nnet.flow_model.x_embedder.modules_to_save['default'].state_dict()
    }
    torch.save(final_training_state, os.path.join(final_save_path, 'training_state.pt'))
    
    absl_logging.info('Training completed!')

def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem
    return 'default_config'

def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        if argv[i].startswith('--config.') and '=' in argv[i]:
            if 'path' in argv[i]: continue
            hparam, val = argv[i].split('=', 1)
            hparam = hparam.split('.')[-1]
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    return hparams if hparams else 'default'

def main(argv):
    config = FLAGS.config
    config.config_name = "configs/WHU.py"
    train(config)

if __name__ == "__main__":
    app.run(main)