"""BuildFlow configuration for the WHU-Mix Dataset.

Fill in every path marked with TODO, then run:

    python train.py --config=configs/WHU_Mix.py [--workdir=/path/to/output]
"""

import os
from dataclasses import dataclass

import ml_collections


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Velocity network input: 512x512 images are encoded by the frozen VAE into
# 64x64x4 latents.
model = Args(latent_size=64, channels=4)


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    # ------------------------------------------------------------------
    # Seed and data pipeline
    # ------------------------------------------------------------------
    # The paper reports mean +/- std over three independent seeds; override
    # on the command line, e.g. --config.seed=42.
    config.seed = 1234
    # Train from the pre-encoded VAE latents produced by extract_features.py.
    config.use_preprocessed = True

    # ------------------------------------------------------------------
    # Output paths (TODO: fill in before running)
    # ------------------------------------------------------------------
    config.workdir = ''
    config.ckpt_root = os.path.join(config.workdir, 'checkpoints')
    config.sample_dir = os.path.join(config.workdir, 'samples')

    # ------------------------------------------------------------------
    # Frozen VAE (CrossFlow assets): encodes 512x512 images/masks into
    # 64x64x4 latents and decodes the final ODE state back to pixels.
    # ------------------------------------------------------------------
    config.autoencoder = d(
        pretrained_path='',     # TODO: path to autoencoder_kl.pth
        scale_factor=0.23010,
    )

    # ------------------------------------------------------------------
    # Velocity field network: DiT-H/2 fine-tuned with LoRA
    # ------------------------------------------------------------------
    config.nnet = d(name='DiT-I2I-H/2', model_args=model)

    # ------------------------------------------------------------------
    # Dataset directories (TODO: fill in before running)
    # All image/mask tiles are 512x512 .tif files, paired by filename stem.
    # ------------------------------------------------------------------
    config.dataset = d(
        name='WHU-Mix',
        image_size=512,
        train_source_root='',   # TODO
        train_target_root='',   # TODO
        val_source_root='',     # TODO (used for best-checkpoint selection)
        val_target_root='',     # TODO
        test_source_root='',    # TODO (official test split, see config.test)
        test_target_root='',    # TODO
        train_latent_root='',   # TODO (output of extract_features.py)
        val_latent_root='',     # TODO (output of extract_features.py)
    )

    # ------------------------------------------------------------------
    # Training schedule (paper Section IV-A3)
    # ------------------------------------------------------------------
    config.train = d(
        n_steps=400000,
        batch_size=4,
        log_interval=1000,
        save_interval=10000,
        ema_rate=0.9999,
        start_eval_step=90000,
        pretrained_ckpt_path='',    # TODO: CrossFlow t2i_512px_t5_dit.pth
        # Initialization of the extra source-injection input channels.
        dvi_init_strategy='copy',
        # Anchor sampling: probability of forcing the source anchor t=0.
        anchor_prob=0.5,
        # Perturbation noise strength for non-anchor states (t > 0).
        sigma_noise=2.0,
    )

    # ------------------------------------------------------------------
    # LoRA fine-tuning: applied to qkv and attn.proj; the x_embedder is
    # fully trained to accept the injected source latent (8 input channels).
    # ------------------------------------------------------------------
    config.lora = d(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=['qkv', 'attn.proj'],
    )

    # ------------------------------------------------------------------
    # Optimizer and learning-rate warmup
    # ------------------------------------------------------------------
    config.optimizer = d(
        name='adamw', lr=1e-4, weight_decay=0.001, betas=(0.9, 0.999)
    )
    config.lr_scheduler = d(name='customized', warmup_steps=5000)

    # ------------------------------------------------------------------
    # Inference: Euler ODE solver with classifier-free guidance
    # ------------------------------------------------------------------
    config.sample = d(
        sample_steps=10,
        scale=2.0,
    )

    # ------------------------------------------------------------------
    # Validation used during training for best-checkpoint selection.
    # For the official test split, point input_dir/label_dir at
    # dataset.test_source_root / dataset.test_target_root and rerun test.py.
    # ------------------------------------------------------------------
    config.test = d(
        ckpt_dir=os.path.join(config.ckpt_root, 'best_model'),
        input_dir=config.dataset.val_source_root,
        output_dir=os.path.join(config.workdir, 'test'),
        label_dir=config.dataset.val_target_root,
        base_model_path=config.train.pretrained_ckpt_path,
        threshold=0.5,
        n_samples=-1,
        sample_steps=10,
        sample_steps_list=None,
        cfg_scale=2.0,
        batch_size=4,
        num_workers=4,
        save_pred_masks=True,
        pred_mask_dir=os.path.join(config.workdir, 'test', 'pred_masks'),
    )
    return config
