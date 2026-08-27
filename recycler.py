"""Diagnosis-conditioned candidate recycling for GEARS."""

import torch

from ma_sde import re_denoise_with_scheduler
from mllm_utils import generate_recycling_prompt
from stage_adapter import get_stage_focus


def renoise_latent(
    clean_latent: torch.Tensor,
    noise_level: float,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Perturb a clean latent to a target noise level."""
    noise = torch.randn(
        clean_latent.shape,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
        generator=generator,
    )
    return (1.0 - noise_level) * clean_latent + noise_level * noise


def recycle_candidate(
    model,
    text_encoder,
    clean_latent: torch.Tensor,
    video_tensor: torch.Tensor,
    current_prompt: str,
    reward_dict: dict,
    checkpoint_noise_level: float,
    checkpoint_idx: int,
    transition_alpha: int,
    scheduler,
    timesteps,
    start_step_idx: int,
    arg_null: dict,
    guide_scale: float,
    eta_sde: float,
    num_train_timesteps: int,
    device: torch.device,
    mllm_model_name: str,
    mllm_base_url: str,
    generator: torch.Generator = None,
    t5_cpu: bool = False,
    offload_model: bool = True,
):
    """Diagnose, re-noise, and re-denoise one recoverable candidate."""
    stage_focus = get_stage_focus(checkpoint_idx, transition_alpha)
    enhanced_prompt = generate_recycling_prompt(
        original_prompt=current_prompt,
        reward_dict=reward_dict,
        video_tensor=video_tensor,
        stage_focus=stage_focus,
        model_name=mllm_model_name,
        base_url=mllm_base_url,
        num_keyframes=4,
    )

    if not t5_cpu:
        text_encoder.model.to(device)
        enhanced_context = text_encoder([enhanced_prompt], device)
        if offload_model:
            text_encoder.model.cpu()
    else:
        enhanced_context = text_encoder(
            [enhanced_prompt],
            torch.device("cpu"),
        )
        enhanced_context = [tensor.to(device) for tensor in enhanced_context]

    arg_c = {
        "context": enhanced_context,
        "seq_len": arg_null["seq_len"],
    }
    noised_latent = renoise_latent(
        clean_latent,
        checkpoint_noise_level,
        generator,
    )
    repaired_latent = re_denoise_with_scheduler(
        model=model,
        noised_latent=noised_latent,
        scheduler=scheduler,
        timesteps=timesteps,
        start_idx=start_step_idx,
        arg_c=arg_c,
        arg_null=arg_null,
        guide_scale=guide_scale,
        eta_sde=eta_sde,
        num_train_timesteps=num_train_timesteps,
        generator=generator,
        device=device,
    )
    return repaired_latent, enhanced_prompt
