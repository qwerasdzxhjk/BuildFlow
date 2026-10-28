"""Training entry point for BuildFlow.

Fine-tunes the frozen CrossFlow DiT with LoRA to learn the velocity field of
the deterministic image-to-mask flow in VAE latent space:

    python train.py --config=configs/WHU.py [--workdir=/path/to/output]
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get(
    'BUILD_FLOW_CUDA_VISIBLE_DEVICES',
    os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
)
import sys
from pathlib import Path
import collections
import random
import numpy as np
import shutil

import ml_collections
import torch
from torch.cuda.amp import GradScaler
from absl import app, flags
from absl import logging as absl_logging
from ml_collections import config_flags
from tqdm.auto import tqdm
import glob
from peft.utils.save_and_load import set_peft_model_state_dict

import utils
import libs.autoencoder
from datasets import create_i2i_dataloaders
from diffusion.flow_matching_loss import ModalityFlowMatching
from diffusion.flow_matching import ODEEulerFlowMatchingSolver

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def _load_peft_adapter(model, adapter_path, device, label):
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Missing {label} adapter checkpoint: {adapter_path}")
    adapter_weights = _load_torch_state(adapter_path, device)
    incompatible = set_peft_model_state_dict(model, adapter_weights, adapter_name='default')
    unexpected = list(getattr(incompatible, 'unexpected_keys', ()))
    if unexpected:
        raise RuntimeError(
            f"{label} adapter has unexpected keys after PEFT remapping: {unexpected[:8]}"
        )
    absl_logging.info(f"Restored {len(adapter_weights)} tensors from {label}: {adapter_path}")


def _capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(rng_state):
    if not rng_state:
        return
    random.setstate(rng_state['python'])
    np.random.set_state(rng_state['numpy'])
    # training_state.pt may be loaded with map_location=device, but RNG APIs
    # require CPU byte tensors even when the model itself lives on CUDA.
    torch_state = torch.as_tensor(rng_state['torch'], dtype=torch.uint8, device='cpu')
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and rng_state.get('cuda') is not None:
        cuda_states = [
            torch.as_tensor(state, dtype=torch.uint8, device='cpu')
            for state in rng_state['cuda']
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def _load_torch_state(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _build_training_state(train_state, scaler, best_iou, config):
    return {
        'step': train_state.step,
        'optimizer': train_state.optimizer.state_dict(),
        'lr_scheduler': train_state.lr_scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_iou': float(best_iou),
        'rng_state': _capture_rng_state(),
        'x_embedder': train_state.nnet.flow_model.x_embedder.modules_to_save['default'].state_dict(),
        'x_embedder_ema': train_state.nnet_ema.flow_model.x_embedder.modules_to_save['default'].state_dict(),
        'seed': int(config.seed),
        'dataset_name': str(config.dataset.name),
    }


def _load_pretrained_base(train_state, config):
    pretrained_ckpt_path = config.train.get('pretrained_ckpt_path')
    if not pretrained_ckpt_path or not os.path.exists(pretrained_ckpt_path):
        raise FileNotFoundError(f"Pre-trained checkpoint not found: {pretrained_ckpt_path}")

    absl_logging.info(f"Loading frozen pre-trained base weights from: {pretrained_ckpt_path}")
    pre_weights = _load_torch_state(pretrained_ckpt_path, 'cpu')
    if 'state_dict' in pre_weights:
        pre_weights = pre_weights['state_dict']
    elif 'model' in pre_weights:
        pre_weights = pre_weights['model']
    elif 'nnet' in pre_weights:
        pre_weights = pre_weights['nnet']

    shared_state = collections.OrderedDict()
    shared_prefixes = ['x_embedder.', 't_embedder.', 'y_embedder.', 'blocks.', 'final_layer.']
    original_x_embedder_weight = None
    original_x_embedder_bias = None

    for key, value in pre_weights.items():
        if key == 'x_embedder.proj.weight':
            original_x_embedder_weight = value
            continue
        if key == 'x_embedder.proj.bias':
            original_x_embedder_bias = value
            continue
        if key == 'pos_embed' or any(key.startswith(prefix) for prefix in shared_prefixes):
            shared_state[f'flow_model.{key}'] = value

    if original_x_embedder_weight is None:
        raise KeyError("Pre-trained checkpoint does not contain x_embedder.proj.weight")

    current_embedder = train_state.nnet.flow_model.x_embedder.modules_to_save['default']
    new_weight = current_embedder.proj.weight.data.clone()
    in_channels_orig = original_x_embedder_weight.shape[1]
    new_weight[:, :in_channels_orig, :, :] = original_x_embedder_weight

    init_strategy = config.train.get('dvi_init_strategy', 'zero')
    if init_strategy == 'random':
        torch.nn.init.kaiming_normal_(new_weight[:, in_channels_orig:, :, :])
    elif init_strategy == 'copy':
        for channel in range(in_channels_orig, new_weight.shape[1]):
            new_weight[:, channel, :, :] = original_x_embedder_weight[:, channel % in_channels_orig, :, :]
    else:
        new_weight[:, in_channels_orig:, :, :].zero_()

    shared_state['flow_model.x_embedder.modules_to_save.default.proj.weight'] = new_weight
    if original_x_embedder_bias is not None:
        shared_state['flow_model.x_embedder.modules_to_save.default.proj.bias'] = original_x_embedder_bias

    train_state.nnet.base_model.model.load_state_dict(shared_state, strict=False)
    train_state.nnet_ema.base_model.model.load_state_dict(shared_state, strict=False)
    absl_logging.info(
        f"Loaded {len(shared_state)} frozen base tensors into both main and EMA models."
    )

def train(config: ml_collections.ConfigDict):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

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
    _load_pretrained_base(train_state, config)
    
    resume_state = None
    did_resume = False
    resume_best_iou = float(config.train.get('resume_best_iou', 0.0))
    latest_ckpt_dir = os.path.join(config.ckpt_root, 'latest')
    legacy_ckpt_dirs = sorted(glob.glob(os.path.join(config.ckpt_root, '[0-9]'*7)))
    if os.path.exists(os.path.join(latest_ckpt_dir, 'training_state.pt')):
        pass
    elif len(legacy_ckpt_dirs) > 0:
        latest_ckpt_dir = legacy_ckpt_dirs[-1]
    else:
        latest_ckpt_dir = None

    if latest_ckpt_dir is not None:
        absl_logging.info(f"Found latest checkpoint: {latest_ckpt_dir}")
        state_path = os.path.join(latest_ckpt_dir, 'training_state.pt')
        if os.path.exists(state_path):
            absl_logging.info(f"Resuming training state from {state_path}")
            state = _load_torch_state(state_path, device)
            saved_seed = state.get('seed')
            saved_dataset = state.get('dataset_name')
            if saved_seed is not None and int(saved_seed) != int(config.seed):
                raise RuntimeError(
                    f"Checkpoint seed {saved_seed} does not match requested seed {config.seed}."
                )
            if saved_dataset is not None and str(saved_dataset) != str(config.dataset.name):
                raise RuntimeError(
                    f"Checkpoint dataset {saved_dataset} does not match requested dataset {config.dataset.name}."
                )
            resume_state = state
            did_resume = True
            train_state.step = state['step']

            lora_main_path = os.path.join(latest_ckpt_dir, 'lora_main', 'adapter_model.bin')
            lora_ema_path = os.path.join(latest_ckpt_dir, 'lora_ema', 'adapter_model.bin')
            _load_peft_adapter(train_state.nnet, lora_main_path, device, 'LoRA main')
            _load_peft_adapter(train_state.nnet_ema, lora_ema_path, device, 'LoRA EMA')

            if 'x_embedder' in state:
                absl_logging.info("Loading fully fine-tuned x_embedder weights from checkpoint.")
                train_state.nnet.flow_model.x_embedder.modules_to_save['default'].load_state_dict(state['x_embedder'])
            else:
                absl_logging.warning("Checkpoint does not contain 'x_embedder' weights. It might be an old checkpoint.")
            if 'x_embedder_ema' in state:
                absl_logging.info("Loading EMA x_embedder weights from checkpoint.")
                train_state.nnet_ema.flow_model.x_embedder.modules_to_save['default'].load_state_dict(state['x_embedder_ema'])

            train_state.optimizer.load_state_dict(state['optimizer'])
            train_state.lr_scheduler.load_state_dict(state['lr_scheduler'])
            resume_best_iou = float(state.get('best_iou', resume_best_iou))
            absl_logging.info(
                f"Resume state restored at step {train_state.step} with best IoU {resume_best_iou:.4f}."
            )
        else:
            raise FileNotFoundError(
                f"Checkpoint directory {latest_ckpt_dir} exists but training_state.pt is missing."
            )
    else:
        absl_logging.info("No checkpoint found, starting from scratch.")
    nnet = train_state.nnet
    nnet_ema = train_state.nnet_ema
    optimizer = train_state.optimizer
    lr_scheduler = train_state.lr_scheduler
    absl_logging.info("Model and optimizer prepared for LoRA fine-tuning.")

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)
    autoencoder.eval()

    anchor_prob = config.train.get('anchor_prob', 0.5)
    sigma_noise = config.train.get('sigma_noise', 2.0)
    flow_matcher = ModalityFlowMatching(
        bbs_prob=anchor_prob,
        sigma_noise=sigma_noise,
    )
    absl_logging.info(
        "VAE and Flow Matcher initialized with anchor_prob=%.3f and sigma_noise=%.3f.",
        anchor_prob,
        sigma_noise,
    )

    scaler = GradScaler()
    if resume_state is not None and 'scaler' in resume_state:
        scaler.load_state_dict(resume_state['scaler'])
        absl_logging.info("AMP GradScaler state restored from checkpoint.")
    if resume_state is not None and 'rng_state' in resume_state:
        _restore_rng_state(resume_state['rng_state'])
        absl_logging.info("Python, NumPy, PyTorch and CUDA RNG states restored from checkpoint.")
    resume_state = None

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

    def train_step(_batch):
        optimizer.zero_grad()
        
        if config.get('use_preprocessed', False):
            source_latent = autoencoder.sample(_batch['source_latent'])
            target_latent = autoencoder.sample(_batch['target_latent'])
        else:
            source_latent = encode(_batch['source_images'])
            target_latent = encode(_batch['target_images'])
            
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            loss, loss_dict = flow_matcher(
                nnet=nnet,
                source_latent=source_latent,
                target_latent=target_latent,
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

    def calculate_iou(pred, label, smooth=1e-6):
        pred = pred.bool()
        label = label.bool()
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
            predicted_masks = generated_gray > threshold

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

    def save_latest_checkpoint(best_iou):
        """Atomically replace the single resumable checkpoint."""
        latest_save_path = os.path.join(config.ckpt_root, 'latest')
        temporary_save_path = os.path.join(config.ckpt_root, '.latest_tmp')
        previous_save_path = os.path.join(config.ckpt_root, '.latest_previous')
        os.makedirs(config.ckpt_root, exist_ok=True)
        if os.path.exists(temporary_save_path):
            shutil.rmtree(temporary_save_path)
        if os.path.exists(previous_save_path):
            shutil.rmtree(previous_save_path)
        os.makedirs(temporary_save_path, exist_ok=True)

        train_state.nnet.save_pretrained(os.path.join(temporary_save_path, 'lora_main'))
        train_state.nnet_ema.save_pretrained(os.path.join(temporary_save_path, 'lora_ema'))
        training_state = _build_training_state(train_state, scaler, best_iou, config)
        torch.save(training_state, os.path.join(temporary_save_path, 'training_state.pt'))

        if os.path.exists(latest_save_path):
            os.replace(latest_save_path, previous_save_path)
        try:
            os.replace(temporary_save_path, latest_save_path)
        except Exception:
            if os.path.exists(previous_save_path) and not os.path.exists(latest_save_path):
                os.replace(previous_save_path, latest_save_path)
            raise
        if os.path.exists(previous_save_path):
            shutil.rmtree(previous_save_path)

        # Remove numbered/final checkpoint layouts used by older versions.
        # best_model is retained separately for testing.
        legacy_paths = glob.glob(os.path.join(config.ckpt_root, '[0-9]' * 7))
        legacy_paths.append(os.path.join(config.ckpt_root, 'final'))
        for legacy_path in legacy_paths:
            if os.path.exists(legacy_path):
                shutil.rmtree(legacy_path)
        absl_logging.info(
            f"Saved the single latest checkpoint at step {train_state.step} to {latest_save_path}"
        )

    def save_best_checkpoint(best_iou):
        """Replace the best-validation checkpoint used for final testing."""
        best_save_path = os.path.join(config.ckpt_root, 'best_model')
        temporary_save_path = os.path.join(config.ckpt_root, '.best_tmp')
        if os.path.exists(temporary_save_path):
            shutil.rmtree(temporary_save_path)
        os.makedirs(temporary_save_path, exist_ok=True)
        train_state.nnet.save_pretrained(os.path.join(temporary_save_path, 'lora_main'))
        train_state.nnet_ema.save_pretrained(os.path.join(temporary_save_path, 'lora_ema'))
        training_state = _build_training_state(train_state, scaler, best_iou, config)
        torch.save(training_state, os.path.join(temporary_save_path, 'training_state.pt'))
        if os.path.exists(best_save_path):
            shutil.rmtree(best_save_path)
        os.replace(temporary_save_path, best_save_path)
        absl_logging.info(f"Updated best checkpoint at {best_save_path}")

    absl_logging.info(f'Start training from step {train_state.step}...')
    pbar = tqdm(initial=train_state.step, total=config.train.n_steps)
    best_iou_so_far = resume_best_iou

    if did_resume and config.train.get('validate_on_resume', False):
        absl_logging.info("Running validation immediately after checkpoint restoration.")
        torch.cuda.empty_cache()
        resume_iou = eval_step(config.sample)
        torch.cuda.empty_cache()
        expected_iou = config.train.get('resume_expected_iou', None)
        if expected_iou is not None:
            iou_delta = abs(float(resume_iou) - float(expected_iou))
            tolerance = float(config.train.get('resume_iou_tolerance', 0.02))
            absl_logging.info(
                f"Resume validation IoU: {resume_iou:.4f}; expected: {expected_iou:.4f}; "
                f"absolute delta: {iou_delta:.4f}."
            )
            if iou_delta > tolerance:
                raise RuntimeError(
                    f"Resume validation delta {iou_delta:.4f} exceeds tolerance {tolerance:.4f}."
                )

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

        if train_state.step % config.train.save_interval == 0:
            start_eval_step = config.train.get('start_eval_step', 0)
            if train_state.step >= start_eval_step:
                torch.cuda.empty_cache()
                eval_iou = eval_step(config.sample)
                torch.cuda.empty_cache()
                if eval_iou > best_iou_so_far:
                    absl_logging.info(
                        f"New best IoU: {best_iou_so_far:.4f} -> {eval_iou:.4f}."
                    )
                    best_iou_so_far = float(eval_iou)
                    save_best_checkpoint(best_iou_so_far)
                else:
                    absl_logging.info(
                        f"Current IoU: {eval_iou:.4f}; best IoU: {best_iou_so_far:.4f}."
                    )
            save_latest_checkpoint(best_iou_so_far)

    pbar.close()
    if train_state.step % config.train.save_interval != 0:
        save_latest_checkpoint(best_iou_so_far)
    absl_logging.info('Training completed!')

def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem
    return 'default_config'

def main(argv):
    config = FLAGS.config
    config.config_name = f"configs/{get_config_name()}.py"
    if FLAGS.workdir:
        config.workdir = FLAGS.workdir
        config.ckpt_root = os.path.join(config.workdir, 'checkpoints')
        config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)

if __name__ == "__main__":
    app.run(main)
