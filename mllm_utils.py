"""MLLM diagnosis and stage-specific prompt generation for Recycler.

Implements the two-regime prompt strategy from the paper:
- High-noise regime (i <= alpha): motion-focused enhancement
- Low-noise regime (i > alpha): visual-focused enhancement
"""
import base64
import io
import os

import torch
from openai import OpenAI
from PIL import Image

MOTION_SYSTEM_PROMPT = (
    "You are a video prompt enhancement specialist. Given an original prompt, "
    "quality scores, and sample frames, you enrich the prompt by adding vivid "
    "motion-related details. Keep ALL original content — only ADD details about "
    "object movement, action trajectory, temporal progression, camera motion, "
    "dynamic interaction, or physically plausible motion patterns. "
    "Output ONLY the enhanced prompt text, nothing else."
)

MOTION_USER_TEMPLATE = """Original prompt: "{original_prompt}"

Video quality scores (higher = better):
{reward_scores}

Weakest dimension: Motion Quality — {weakest_dim_description}

Suggested enhancement directions: {enhancement_hints}

I have attached {num_frames} sample frames from the current video.

Enrich the original prompt by adding vivid motion-related details that would improve the weakest dimension. Keep ALL original content — only ADD details about object movement, action trajectory, temporal progression, camera motion, dynamic interaction, or physically plausible motion patterns. The enhanced prompt should repair the motion weakness while preserving the existing subject, scene layout, and visual identity shown in the frames.

Output ONLY the enhanced prompt text, nothing else."""

VISUAL_SYSTEM_PROMPT = (
    "You are a video prompt enhancement specialist. Given an original prompt, "
    "quality scores, and sample frames, you enrich the prompt by adding vivid "
    "visual details. Keep ALL original content — only ADD scene, lighting, camera, "
    "texture, material, color, atmosphere, composition, local-detail, or style details. "
    "Output ONLY the enhanced prompt text, nothing else."
)

VISUAL_USER_TEMPLATE = """Original prompt: "{original_prompt}"

Video quality scores (higher = better):
{reward_scores}

Weakest dimension: Visual Quality — {weakest_dim_description}

Suggested enhancement directions: {enhancement_hints}

I have attached {num_frames} sample frames from the current video.

Enrich the original prompt by adding vivid visual details that would improve the weakest dimension. Keep ALL original content — only ADD scene, lighting, camera, texture, material, color, atmosphere, composition, local-detail, or style details. The enhanced prompt should repair the visual weakness while preserving the existing subject, layout, motion, and semantic content shown in the frames.

Output ONLY the enhanced prompt text, nothing else."""

MOTION_ENHANCEMENT_HINTS = (
    "smooth and fluid camera movement, dynamic yet stable motion, "
    "natural and physically plausible animation, cinematic camera panning or tracking, "
    "vivid action dynamics and temporal progression"
)

VISUAL_ENHANCEMENT_HINTS = (
    "vivid lighting and color palette, high-definition textures and sharp details, "
    "cinematic composition and depth of field, rich contrast and visual clarity, "
    "atmospheric effects and material fidelity"
)


def extract_keyframes_as_base64(video_tensor, num_frames=4):
    """Extract evenly-spaced keyframes from video tensor (C, T, H, W) as base64 JPEGs."""
    total_frames = video_tensor.shape[1]
    indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    base64_frames = []
    for idx in indices:
        frame = video_tensor[:, idx, :, :]
        frame = ((frame + 1) / 2 * 255).clamp(0, 255).byte()
        frame = frame.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(frame)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        base64_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return base64_frames


def generate_recycling_prompt(
    original_prompt: str,
    reward_dict: dict,
    video_tensor: torch.Tensor,
    stage_focus: str,
    model_name: str,
    base_url: str,
    num_keyframes: int = 4,
) -> str:
    """Generate a diagnosis-conditioned enhanced prompt with an MLLM."""
    if not model_name:
        raise ValueError("model_name must be provided.")
    if not base_url:
        raise ValueError("base_url must be provided.")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure it in the runtime environment."
        )

    reward_scores_text = (
        f"  - Visual Quality (VQ): {reward_dict['VQ']:.4f}\n"
        f"  - Motion Quality (MQ): {reward_dict['MQ']:.4f}\n"
        f"  - Text Alignment (TA): {reward_dict['TA']:.4f}"
    )

    if stage_focus == "MQ":
        system_prompt = MOTION_SYSTEM_PROMPT
        user_template = MOTION_USER_TEMPLATE
        weakest_desc = "temporal consistency, motion smoothness, and natural movement"
        hints = MOTION_ENHANCEMENT_HINTS
    else:
        system_prompt = VISUAL_SYSTEM_PROMPT
        user_template = VISUAL_USER_TEMPLATE
        weakest_desc = "clearness, resolution, brightness, color, and overall visual appeal"
        hints = VISUAL_ENHANCEMENT_HINTS

    user_text = user_template.format(
        original_prompt=original_prompt,
        reward_scores=reward_scores_text,
        weakest_dim_description=weakest_desc,
        enhancement_hints=hints,
        num_frames=num_keyframes,
    )

    keyframe_base64_list = extract_keyframes_as_base64(
        video_tensor,
        num_keyframes,
    )

    content_parts = [{"type": "text", "text": user_text}]
    for frame_b64 in keyframe_base64_list:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "low"},
        })

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        enhanced = response.choices[0].message.content.strip().strip('"').strip("'")
        if len(enhanced) > 500:
            enhanced = enhanced[:500]
        return enhanced
    except Exception as exc:
        raise RuntimeError(f"MLLM request failed: {exc}") from exc
