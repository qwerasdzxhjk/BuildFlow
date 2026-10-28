import torch
import torch.nn as nn


class ModalityFlowMatching(nn.Module):
    """
    Flow-matching objective for direct modality-to-modality evolution.

    1. Anchor-aware time sampling: with probability ``bbs_prob`` the exact
       source anchor ``t=0`` is selected, otherwise ``t`` is uniform in [0, 1).
    2. A stochastic interpolant that injects noise of strength ``sigma_noise``
       into non-anchor states (``t > 0``).
    3. The flow-matching loss: MSE between the predicted velocity and the
       constant ideal velocity (z1 - z0).
    """

    def __init__(self, bbs_prob: float = 0.5, sigma_noise: float = 2.0):
        super().__init__()
        self.bbs_prob = bbs_prob
        self.time_step_sampler = self._boundary_biased_sampler
        self.sigma_noise = sigma_noise

    def _boundary_biased_sampler(self, x: torch.Tensor) -> torch.Tensor:
        """Sample the anchor-aware time schedule."""
        bsz = x.shape[0]
        device = x.device

        if not 0.0 <= self.bbs_prob <= 1.0:
            raise ValueError(f"bbs_prob must be in [0, 1], got {self.bbs_prob}")

        anchor_mask = torch.rand(bsz, device=device) < self.bbs_prob
        uniform_t = torch.rand(bsz, device=device)
        return torch.where(anchor_mask, torch.zeros_like(uniform_t), uniform_t)

    def mos(self, err, start_dim=1):
        """Mean of Squares helper function."""
        return err.pow(2).flatten(start_dim=start_dim).mean(dim=-1)

    def forward(self, nnet, source_latent, target_latent):
        """Compute the flow-matching loss (MSE on the velocity)."""
        t = self.time_step_sampler(target_latent)
        z0 = source_latent
        z1 = target_latent  # Ground truth target latent

        # Generate the noisy sample using the stochastic path function.
        x_noisy = self.psi(t, x=z0, x1=z1)

        # The target velocity is the clean-path velocity.
        target_velocity = self.Dt_psi(t, x=z0, x1=z1)

        # Predict velocity from the noisy input and timestep.
        prediction_tuple = nnet(x_noisy, t, z0, null_indicator=None)
        predicted_velocity = prediction_tuple[0]

        # Flow-matching loss: L2 loss on velocity.
        loss_diff = self.mos(predicted_velocity - target_velocity)
        loss = loss_diff.mean()

        # Logging dictionary for monitoring.
        loss_dict = {"loss_diff": loss_diff.mean().detach()}
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
