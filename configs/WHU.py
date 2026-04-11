import ml_collections
import os
from dataclasses import dataclass

@dataclass
class Args:
    """A simple dataclass-like helper to store nested configurations."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# =============================================================================
# == DiT Model Arguments (Defined globally, similar to the reference) ==
# =============================================================================
# NOTE: Customize these arguments to match exactly what your I2I DiT model needs.
# The values here are tailored for I2I, removing text-specific ones like 'clip_dim'.
model = Args(
    # Core DiT architecture arguments
    latent_size = 64,
    channels = 4,
)


def d(**kwargs):
    """Helper function to create a ConfigDict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    """Get the hyperparameter configuration for I2I Flow Matching training."""
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.use_preprocessed = True

    # --- Path Settings ---
    # TODO: Fill in your local work directory before training.
    config.workdir = ""
    config.ckpt_root = os.path.join(config.workdir, 'checkpoints')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    
    # --- Component Configurations ---
    config.autoencoder = d(
        # TODO: Fill in your local VAE checkpoint path.
        pretrained_path="",
        scale_factor=0.23010
    )

    # --- Use the globally defined model arguments ---
    global model
    config.nnet = d(
        name='DiT-I2I-H/2', # The key for your model in DiT_I2I_models dict
        model_args=model,   # Assigning the global 'model' object here
    )

    config.dataset = d(
        image_size=512,
        # TODO: Fill in your local training image directory.
        train_source_root='',
        # TODO: Fill in your local training mask directory.
        train_target_root='',
        # TODO: Fill in your local validation image directory.
        val_source_root='',
        # TODO: Fill in your local validation mask directory.
        val_target_root='',
        # TODO: Fill in your local preprocessed training latent directory.
        train_latent_root='',
        # TODO: Fill in your local preprocessed validation latent directory.
        val_latent_root='',
        # TODO: Fill in your local test image directory.
        test_source_root='',
        # TODO: Fill in your local test mask directory.
        test_target_root='',
    )
    
    # --- Training Loop & Hyperparameters ---
    config.train = d(
        n_steps=200000,
        batch_size=4,
        num_workers=8,
        log_interval=1000,
        save_interval=10000,
        ema_rate=0.9999,
        max_checkpoints_to_keep=10,
        # TODO: Fill in your local pretrained DiT checkpoint path.
        pretrained_ckpt_path="",
        start_eval_step = 90000,
        dvi_init_strategy = 'copy',
        # 'zero': initialize new channels with zeros (default)
        # 'random': initialize new channels with Kaiming Normal
        # 'copy': copy original pretrained channel weights into new channels
    )

    config.lora = d(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=["qkv", "attn.proj"] # Adjust according to your model definition
    )

    config.optimizer = d(
        name='adamw',
        lr=1e-4,
        weight_decay=0.001,
        betas=(0.9, 0.999)
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    # --- Sampling / Evaluation ---
    config.sample = d(
        sample_steps=10,
        n_samples=16,
        scale=2.0,
        mini_batch_size=4
    )

    config.test = d(
        ckpt_dir=os.path.join(config.ckpt_root, 'best_model'),
        input_dir=config.dataset.test_source_root,
        output_dir=os.path.join(config.workdir, 'test'),
        label_dir=config.dataset.test_target_root,
        base_model_path=config.train.pretrained_ckpt_path,
        threshold=0.5,
        n_samples=-1,
        sample_steps=config.sample.sample_steps,
        sample_steps_list=None,
        cfg_scale=config.sample.scale,
        batch_size=4,
        num_workers=4,
        save_pred_masks=True,
        pred_mask_dir=os.path.join(config.workdir, 'test', 'pred_masks')
    )
    return config
