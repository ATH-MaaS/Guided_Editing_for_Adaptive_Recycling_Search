#!/usr/bin/env bash

export OPENAI_API_KEY=xxx

python -s "$(dirname "$0")/generate_vbench_wan_gears.py" \
    --ckpt_dir "/path/to/Wan2.1-T2V-1.3B" \
    --reward_model_path "/path/to/VideoReward" \
    --prompt_dir "/path/to/VBench/prompts/prompts_per_dimension" \
    --output_dir "./outputs/vbench" \
    --mllm_model_name "qwen3.5-plus" \
    --mllm_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --task "t2v-1.3B" \
    --num_gpus 1 \
    --size "832*480" \
    --frame_num 33 \
    --sample_steps 50 \
    --sample_shift 5.0 \
    --sample_guide_scale 5.0 \
    --sample_solver "dpm++" \
    --videos_per_prompt 5 \
    --checkpoints 1.0 0.6 0.3 \
    --transition_alpha 1 \
    --initial_population 6 \
    --valid_candidates 4 \
    --editing_ratio 0.5 \
    --eta_sde 0.5
