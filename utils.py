"""Training state construction (model, LoRA, optimizer, scheduler) and
small shared helpers used by the training and evaluation scripts."""

import torch.nn as nn
from absl import logging
from libs.dit import DiT_I2I_models
from peft import LoraConfig, get_peft_model


def set_logger(log_level="info", fname=None):
    import logging as _logging

    handler = logging.get_absl_handler()
    formatter = _logging.Formatter("%(asctime)s - %(filename)s - %(message)s")
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def get_nnet(name, **kwargs):
    if name in DiT_I2I_models:
        model_class = DiT_I2I_models[name]
        return model_class(kwargs["model_args"])
    else:
        raise NotImplementedError(name)


def get_optimizer(params, name, **kwargs):
    if name == "adam":
        from torch.optim import Adam

        return Adam(params, **kwargs)
    elif name == "adamw":
        from torch.optim import AdamW

        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR

    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1

    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == "customized":
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def initialize_train_state(config, device, layers_to_fully_train=None):
    """
    Initialize training state, including model, optimizer, and learning-rate scheduler.
    Supports hybrid fine-tuning setups (LoRA + fully trained modules).

    Args:
        config: Training configuration.
        device: Target device.
        layers_to_fully_train (list, optional): Prefixes of layers to unfreeze for
            full-parameter training (passed to PEFT ``modules_to_save``).
    """
    nnet = get_nnet(**config.nnet)
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()

    if config.get("lora") is not None:
        logging.info("LoRA config found, applying PEFT to the model...")
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias="none",
            modules_to_save=layers_to_fully_train
        )
        nnet = get_peft_model(nnet, lora_config)
        # Apply the LoRA structure to the EMA model as well so weights can be copied correctly.
        nnet_ema = get_peft_model(nnet_ema, lora_config)
        logging.info("PEFT LoRA applied successfully.")
        nnet.print_trainable_parameters()

    # Keep the frozen base weights of fully-trained modules out of the optimizer.
    for name, param in nnet.named_parameters():
        if ".original_module." in name:
            param.requires_grad = False

    logging.info("Creating optimizer for all trainable parameters...")
    trainable_params = filter(lambda p: p.requires_grad, nnet.parameters())

    num_trainable = 0
    for name, param in nnet.named_parameters():
        if param.requires_grad:
            logging.info(f"  - Trainable: {name} | Size: {param.shape}")
            num_trainable += param.numel()
    logging.info(f"Total number of trainable parameters: {num_trainable / 1e6:.2f}M")

    optimizer = get_optimizer(trainable_params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        step=0,
        nnet=nnet,
        nnet_ema=nnet_ema,
    )
    train_state.ema_update(0)  # Ensure nnet_ema starts from the same weights as nnet.
    train_state.to(device)

    return train_state
