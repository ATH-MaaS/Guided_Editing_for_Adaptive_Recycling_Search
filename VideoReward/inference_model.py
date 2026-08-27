"""Inference-only VideoReward model construction.

This module keeps standard GEARS inference independent of VideoReward's training
stack and its TRL dependency.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    """Qwen2-VL with a token-level multi-dimensional reward head."""

    def __init__(
        self,
        config,
        output_dim=4,
        reward_token="last",
        special_token_ids=None,
    ):
        super().__init__(config)
        self.output_dim = output_dim
        self.rm_head = nn.Linear(
            config.text_config.hidden_size,
            output_dim,
            bias=False,
        )
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        del labels, rope_deltas, kwargs
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.get_dtype())
                image_embeds = self.model.visual(
                    pixel_values,
                    grid_thw=image_grid_thw,
                )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                image_embeds = image_embeds.to(
                    inputs_embeds.device,
                    inputs_embeds.dtype,
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask,
                    image_embeds,
                )

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(
                    self.model.visual.get_dtype()
                )
                video_embeds = self.model.visual(
                    pixel_values_videos,
                    grid_thw=video_grid_thw,
                ).last_hidden_state
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                video_embeds = video_embeds.to(
                    inputs_embeds.device,
                    inputs_embeds.dtype,
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    video_mask,
                    video_embeds,
                )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.rm_head(outputs[0])
        batch_size = (
            input_ids.shape[0]
            if input_ids is not None
            else inputs_embeds.shape[0]
        )

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes greater than one without a pad token."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = (
                torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            )
            sequence_lengths %= input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

        if self.reward_token == "last":
            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device),
                sequence_lengths,
            ]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(
                sequence_lengths,
                min=0,
                max=logits.size(1) - 1,
            )
            pooled_logits = torch.stack([
                logits[index, : valid_lengths[index]].mean(dim=0)
                for index in range(batch_size)
            ])
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask |= input_ids == special_token_id
            pooled_logits = logits[special_token_mask].view(batch_size, 3, -1)
            if self.output_dim == 3:
                pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
            pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            raise ValueError(f"Unsupported reward token: {self.reward_token}")

        return {"logits": pooled_logits}


def _find_target_linear_names(
    model,
    num_lora_modules=-1,
    excluded_names=None,
):
    excluded_names = excluded_names or []
    module_names = []
    for name, module in model.named_modules():
        if any(excluded in name for excluded in excluded_names):
            continue
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module_names.append(name)
    if num_lora_modules > 0:
        module_names = module_names[-num_lora_modules:]
    return module_names


def create_model_and_processor(
    model_config,
    peft_lora_config,
    training_args,
    cache_dir=None,
):
    """Build the VideoReward inference model without importing TRL."""
    if model_config.load_in_8bit or model_config.load_in_4bit:
        raise ValueError(
            "Quantized VideoReward loading is not supported by this inference entry."
        )

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in {"auto", None}
        else getattr(torch, model_config.torch_dtype)
    )
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        padding_side="right",
        cache_dir=cache_dir,
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = [
            "<|VQ_reward|>",
            "<|MQ_reward|>",
            "<|TA_reward|>",
        ]
        processor.tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(
            special_tokens
        )

    model = Qwen2VLRewardModelBT.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    )
    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    peft_config = None
    if peft_lora_config.lora_enable:
        target_modules = _find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            excluded_names=peft_lora_config.lora_namespan_exclude,
        )
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    return model, processor, peft_config

