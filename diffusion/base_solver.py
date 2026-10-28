"""
This file contains the ODE solver base class with the classifier-free guidance
(null-indicator) used by BuildFlow inference.
"""

import torch


class Solver:
    def __init__(self, model_fn, **kwargs):
        self.model = model_fn

    @torch.no_grad()
    def sample(self, sampling_method, unconditional_guidance_scale, has_null_indicator, x_T=None, **kwargs):
        # Remaining kwargs (e.g. batch_size, sample_steps) are consumed by the
        # concrete solver before reaching this method.
        samples, intermediates = sampling_method(
            x_T=x_T,
            unconditional_guidance_scale=unconditional_guidance_scale,
            has_null_indicator=has_null_indicator,
        )
        return samples, intermediates

    @torch.no_grad()
    def get_model_output_dimr(
        self,
        x,
        t_continuous,
        unconditional_guidance_scale,
        has_null_indicator,
    ):
        z0 = self.source_latent
        log_snr = 4 - t_continuous * 8  # inversed

        if has_null_indicator:
            _cond = self.model(x, t=t_continuous, source_latent=z0, log_snr=log_snr, null_indicator=torch.tensor([False] * x.shape[0]).to(x.device))[-1]
            _uncond = self.model(x, t=t_continuous, source_latent=z0, log_snr=log_snr, null_indicator=torch.tensor([True] * x.shape[0]).to(x.device))[-1]

            assert unconditional_guidance_scale > 1
            return _uncond + unconditional_guidance_scale * (_cond - _uncond)
        else:
            _cond = self.model(x, log_snr=log_snr)[-1]
            return _cond
