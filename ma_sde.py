"""Manifold-aware SDE sampling used by Recycler."""

import math

import torch


def compute_ma_sde_noise_std(
    noise_level_current: float,
    noise_level_next: float,
    eta_sde: float,
) -> float:
    """Compute the injected noise standard deviation for one solver step."""
    delta = noise_level_current - noise_level_next
    log_ratio = math.log(
        (1.0 - noise_level_next) / (1.0 - noise_level_current)
    )
    variance = -delta + log_ratio
    return eta_sde * math.sqrt(variance) if variance > 0 else 0.0


def re_denoise_with_scheduler(
    model,
    noised_latent: torch.Tensor,
    scheduler,
    timesteps,
    start_idx: int,
    arg_c: dict,
    arg_null: dict,
    guide_scale: float,
    eta_sde: float = 0.5,
    num_train_timesteps: int = 1000,
    generator: torch.Generator = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Re-denoise a latent with scheduler steps and MA-SDE noise injection."""
    latent = noised_latent.clone()
    if latent.dim() == 5:
        latent = latent.squeeze(0)

    denoising_timesteps = timesteps[start_idx:]
    scheduler._step_index = None
    scheduler.lower_order_nums = 0
    scheduler.model_outputs = [None] * scheduler.config.solver_order

    for step_idx, timestep in enumerate(denoising_timesteps):
        latent_input = [latent]
        timestep_tensor = torch.stack([timestep]).to(device)
        noise_pred_cond = model(latent_input, t=timestep_tensor, **arg_c)[0]
        noise_pred_uncond = model(latent_input, t=timestep_tensor, **arg_null)[0]
        noise_pred = noise_pred_uncond + guide_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        latent = scheduler.step(
            noise_pred.unsqueeze(0),
            timestep,
            latent.unsqueeze(0),
            return_dict=False,
            generator=generator,
        )[0].squeeze(0)

        if step_idx < len(denoising_timesteps) - 1:
            current_level = float(timestep) / num_train_timesteps
            next_level = (
                float(denoising_timesteps[step_idx + 1]) / num_train_timesteps
            )
            noise_std = compute_ma_sde_noise_std(
                current_level,
                next_level,
                eta_sde,
            )
            if noise_std > 0:
                noise = torch.randn(
                    latent.shape,
                    device=device,
                    generator=generator,
                    dtype=latent.dtype,
                )
                latent = latent + noise_std * noise

    return latent
