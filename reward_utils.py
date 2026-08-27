"""Multi-dimensional reward utilities for GEARS."""

import torch

from VideoReward.score import VideoVLMRewardInference


def get_multidim_reward(
    verifier: VideoVLMRewardInference,
    video_tensor: torch.Tensor,
    prompt: str,
) -> dict:
    """Return the VideoReward VQ, MQ, and TA scores for one video."""
    with torch.no_grad():
        result = verifier.reward(
            [video_tensor.permute(1, 0, 2, 3)],
            [prompt],
            use_norm=True,
        )
    return result[0]
