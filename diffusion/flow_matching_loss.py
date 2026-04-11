import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityFlowMatching(nn.Module):
    """
    A generic flow-matching module for direct modality-to-modality evolution.

    This version includes:
    1. Boundary-Biased Sampling (BBS) for sampling timestep t.
    2. A stochastic interpolant that injects time-dependent noise into the
       linear path to improve robustness.
    3. Flow-matching loss (MSE on velocity) together with an optional
       pixel-space Dice reconstruction loss.
    """

    def __init__(
        self,
        sigma_min: float = 1e-5,
        sigma_max: float = 1.0,
        bbs_prob: float = 1.0,       # Controls how often t=0 is sampled
        sigma_noise: float = 2.0,    # Controls the injected noise strength
    ):
        super().__init__()
        self.bbs_prob = bbs_prob
        self.time_step_sampler = self._boundary_biased_sampler
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_noise = sigma_noise  # Store noise strength

    def _boundary_biased_sampler(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample timestep t using the Boundary-Biased Sampling (BBS) strategy.
        """
        bsz = x.shape[0]
        device = x.device
        random_draw = torch.rand(bsz, device=device)
        uniform_t = torch.rand(bsz, device=device)
        t = torch.where(random_draw < self.bbs_prob, torch.zeros_like(uniform_t), uniform_t)
        return t

    def mos(self, err, start_dim=1):
        """Mean of Squares helper function."""
        return err.pow(2).flatten(start_dim=start_dim).mean(dim=-1)

    def dice_loss(self, pred, target, smooth=1e-5):
        """Calculate Dice loss."""
        # Flatten tensors.
        pred = pred.view(-1)
        target = target.view(-1)

        # Compute the intersection.
        intersection = (pred * target).sum()

        # Compute the Dice coefficient.
        dice_coeff = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        # Return Dice loss.
        return 1. - dice_coeff

    def forward(
        self,
        nnet,
        source_latent,
        target_latent,
        target_mask,  # Target segmentation mask
        autoencoder,  # Autoencoder used for decoding
        all_config=None,
        lambda_recon: float = 1.0,  # Weight of the reconstruction loss
    ):
        """
        Compute the combined flow-matching loss and pixel-space Dice loss.
        """
        # --- 1. Core flow-matching computation ---
        t = self.time_step_sampler(target_latent)
        z0 = source_latent
        z1 = target_latent  # Ground truth target latent

        # Generate the noisy sample using the stochastic path function.
        x_noisy = self.psi(t, x=z0, x1=z1)

        # The target velocity remains the clean-path velocity.
        target_velocity = self.Dt_psi(t, x=z0, x1=z1)

        # Predict velocity from the noisy input and timestep.
        prediction_tuple = nnet(x_noisy, t, z0, null_indicator=None)
        predicted_velocity = prediction_tuple[0]

        # Flow-matching loss: L2 loss on velocity.
        loss_diff = self.mos(predicted_velocity - target_velocity)

        # --- 2. Pixel-space reconstruction loss (Dice loss) ---
        # A. Estimate the target latent z1_hat from z0 and the predicted velocity.
        #    This mimics a single ODE solver step: start from z0, take one step
        #    with the predicted velocity, and obtain the final z1 estimate.
        # predicted_z1 = z0 + predicted_velocity

        # B. Decode the predicted latent back to pixel space as a 3-channel image.
        # decoded_image = autoencoder.decode(predicted_z1)

        # C. Average across the channel dimension to obtain a single-channel mask.
        # predicted_mask_logits = decoded_image.mean(dim=1, keepdim=True)

        # D. Apply sigmoid to map logits into [0, 1] probabilities.
        # predicted_mask_prob = torch.sigmoid(predicted_mask_logits)

        # E. Compute Dice loss between the predicted probabilities and the target mask.
        # loss_recon_dice = self.dice_loss(predicted_mask_prob, target_mask)

        # --- 3. Combine losses ---
        # Add flow-matching loss and pixel-space reconstruction loss.
        # loss = loss_diff.mean() + lambda_recon * loss_recon_dice
        loss = loss_diff.mean()

        # Logging dictionary for monitoring.
        loss_dict = {
            "loss_diff": loss_diff.mean().detach(),
            # "loss_recon_dice": loss_recon_dice.detach(),
        }

        return loss, loss_dict

    def psi(self, t, x, x1):
        """
        Generate a sample point x_t on the stochastic interpolation path.

        x_t = (1 - t) * z0 + t * z1 + mask * sigma * epsilon
        mask: 1 when t > 0, otherwise 0.
        """
        t_expanded = self.expand_t(t, x)

        # 1. Compute the path mean, i.e. the clean linear interpolation.
        mean_path = (1 - t_expanded) * x + t_expanded * x1

        # 2. Compute the noise standard deviation.
        noise_std = self.sigma_noise

        # 3. Generate Gaussian noise with the same shape as the input.
        noise = torch.randn_like(x)

        # 4. Create a mask so noise is added only when t > 0.
        # (t > 0) produces a boolean tensor; .float() maps it to 0.0 or 1.0.
        mask = (t > 0).float()
        # Expand the mask dimensions to match x.
        mask_expanded = self.expand_t(mask, x)

        # 5. Return the final sample, adding noise only when t > 0.
        return mean_path + noise_std * mask_expanded * noise

    def Dt_psi(self, t: torch.Tensor, x: torch.Tensor, x1: torch.Tensor):
        # The target velocity remains the clean-path velocity.
        return x1 - x

    def expand_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand timestep tensor t to match the dimensionality of tensor x."""
        t_expanded = t
        while t_expanded.ndim < x.ndim:
            t_expanded = t_expanded.unsqueeze(-1)
        return t_expanded.expand_as(x)
