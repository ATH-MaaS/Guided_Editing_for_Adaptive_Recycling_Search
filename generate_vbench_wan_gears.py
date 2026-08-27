"""Generate Wan2.1-T2V videos for the standard VBench protocol with GEARS."""

import argparse
import logging
import os
import queue
import sys
from pathlib import Path

import torch.multiprocessing as mp


SUPPORTED_SIZES = {
    "t2v-1.3B": ["832*480"],
    "t2v-14B": ["1280*720", "832*480"],
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GEARS generation for the standard VBench prompt set."
    )
    parser.add_argument(
        "--task",
        default="t2v-1.3B",
        choices=["t2v-1.3B", "t2v-14B"],
    )
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--reward_model_path", required=True)
    parser.add_argument("--prompt_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs/vbench")
    parser.add_argument("--mllm_model_name", required=True)
    parser.add_argument("--mllm_base_url", required=True)

    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=33)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument(
        "--sample_solver",
        default="dpm++",
        choices=["unipc", "dpm++"],
    )
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--offload_model", type=str_to_bool, default=True)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1)

    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--videos_per_prompt", type=int, default=5)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
    )
    parser.add_argument("--save_intermediates", action="store_true")

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=float,
        default=[1.0, 0.6, 0.3],
    )
    parser.add_argument("--transition_alpha", type=int, default=1)
    parser.add_argument("--initial_population", type=int, default=6)
    parser.add_argument("--valid_candidates", type=int, default=4)
    parser.add_argument("--editing_ratio", type=float, default=0.5)
    parser.add_argument("--eta_sde", type=float, default=0.5)

    args = parser.parse_args()
    if args.size not in SUPPORTED_SIZES[args.task]:
        parser.error(
            f"Size {args.size!r} is not supported for {args.task}. "
            f"Choose from {SUPPORTED_SIZES[args.task]}."
        )
    if args.num_gpus < 1:
        parser.error("--num_gpus must be at least 1.")
    if args.videos_per_prompt < 1:
        parser.error("--videos_per_prompt must be at least 1.")
    if args.initial_population < 1:
        parser.error("--initial_population must be at least 1.")
    if args.valid_candidates < 1:
        parser.error("--valid_candidates must be at least 1.")
    if args.valid_candidates > args.initial_population:
        parser.error("--valid_candidates cannot exceed --initial_population.")
    if not 0.0 <= args.editing_ratio <= 1.0:
        parser.error("--editing_ratio must be in [0, 1].")
    if len(args.checkpoints) < 2:
        parser.error("--checkpoints must contain at least two noise levels.")
    if any(level < 0.0 or level > 1.0 for level in args.checkpoints):
        parser.error("Every checkpoint noise level must be in [0, 1].")
    if any(
        current <= following
        for current, following in zip(args.checkpoints, args.checkpoints[1:])
    ):
        parser.error("--checkpoints must be strictly decreasing.")
    if not 1 <= args.transition_alpha < len(args.checkpoints):
        parser.error(
            "--transition_alpha must index one of the recycling checkpoints."
        )
    return args


def collect_tasks(args):
    prompt_dir = Path(args.prompt_dir)
    if not prompt_dir.is_dir():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    if args.categories:
        prompt_files = [prompt_dir / f"{name}.txt" for name in args.categories]
    else:
        prompt_files = sorted(prompt_dir.glob("*.txt"))

    model_name = Path(args.ckpt_dir).name
    tasks = []
    for prompt_file in prompt_files:
        if not prompt_file.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        category = prompt_file.stem
        category_dir = Path(args.output_dir) / model_name / category
        category_dir.mkdir(parents=True, exist_ok=True)
        prompts = [
            line.strip()
            for line in prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.max_prompts is not None:
            prompts = prompts[: args.max_prompts]

        for prompt in prompts:
            for video_index in range(args.videos_per_prompt):
                save_path = category_dir / f"{prompt}-{video_index}.mp4"
                if args.skip_existing and save_path.exists():
                    continue
                tasks.append(
                    {
                        "category": category,
                        "prompt": prompt,
                        "video_index": video_index,
                        "save_path": str(save_path),
                        "seed": args.base_seed + video_index,
                    }
                )
    return tasks


def worker(gpu_id, task_queue, args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import torch

    from VideoReward.score import VideoVLMRewardInference
    from gears_wan_t2v import WanT2VGEARS
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import cache_video

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    logging.basicConfig(level=logging.INFO, format=f"[GPU {gpu_id}] %(message)s")

    config = WAN_CONFIGS[args.task]
    video_size = SIZE_CONFIGS[args.size]
    output_fps = args.fps or config.sample_fps
    pipeline = WanT2VGEARS(
        config=config,
        checkpoint_dir=args.ckpt_dir,
        device_id=gpu_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=args.t5_cpu,
    )
    verifier = VideoVLMRewardInference(
        args.reward_model_path,
        device=device,
        dtype=torch.float16,
    )

    while True:
        try:
            task = task_queue.get(timeout=10)
        except queue.Empty:
            break

        try:
            video = pipeline.generate(
                input_prompt=task["prompt"],
                size=video_size,
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=task["seed"],
                offload_model=args.offload_model,
                checkpoints=tuple(args.checkpoints),
                transition_alpha=args.transition_alpha,
                initial_population=args.initial_population,
                valid_candidates=args.valid_candidates,
                editing_ratio=args.editing_ratio,
                eta_sde=args.eta_sde,
                verifier=verifier,
                mllm_model_name=args.mllm_model_name,
                mllm_base_url=args.mllm_base_url,
            )
            if video is None:
                logging.warning("No output for prompt %r.", task["prompt"])
                continue

            cache_video(
                tensor=video[None],
                save_file=task["save_path"],
                fps=output_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            logging.info("Saved %s", task["save_path"])

            if args.save_intermediates:
                intermediate_dir = Path(task["save_path"]).with_suffix("")
                intermediate_dir = Path(f"{intermediate_dir}_intermediates")
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                for entry in pipeline.intermediate_videos:
                    reward_suffix = f"_r{entry['reward']:.4f}"
                    intermediate_path = intermediate_dir / (
                        f"{entry['label']}{reward_suffix}.mp4"
                    )
                    cache_video(
                        tensor=entry["video"][None],
                        save_file=str(intermediate_path),
                        fps=output_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )
        except Exception:
            logging.exception("Generation failed for prompt %r.", task["prompt"])


def main():
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure it before running this script."
        )

    tasks = collect_tasks(args)
    if not tasks:
        print("No pending generation tasks.")
        return

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    for task in tasks:
        task_queue.put(task)

    workers = []
    for gpu_id in range(min(args.num_gpus, len(tasks))):
        process = context.Process(
            target=worker,
            args=(gpu_id, task_queue, args),
        )
        process.start()
        workers.append(process)

    for process in workers:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker process {process.pid} exited with code {process.exitcode}."
            )

    print(f"Generated videos are available under {args.output_dir}.")


if __name__ == "__main__":
    main()
