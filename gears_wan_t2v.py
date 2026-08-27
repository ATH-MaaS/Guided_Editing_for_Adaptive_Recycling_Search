"""GEARS pipeline for Wan2.1-T2V.

Implements the multi-checkpoint generation-evaluation-editing loop:
  For each checkpoint s_i:
    1. Generate: denoise candidates from s_i to 0, decode to video
    2. Evaluate: multi-dim reward scoring (VQ, MQ, TA)
    3. StageAdapter: route into elites, recyclable candidates, and discards
    4. Recycler: repair recyclable candidates with MLLM diagnosis and MA-SDE
    5. Merge survivors → re-noise to s_{i+1} for next checkpoint
"""
import gc
import math
import random
import sys
from contextlib import contextmanager

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from wan.text2video import WanT2V
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from recycler import recycle_candidate, renoise_latent
from reward_utils import get_multidim_reward
from stage_adapter import get_stage_focus, route_candidates


class WanT2VGEARS(WanT2V):
    """GEARS for Wan2.1-T2V."""

    def _compute_target_shape(self, size, frame_num):
        return (
            self.vae.model.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )

    def _compute_seq_len(self, target_shape):
        return math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        ) * self.sp_size

    def _build_scheduler(self, sample_solver, sampling_steps, shift):
        """Build a single scheduler and return (scheduler, timesteps)."""
        if sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                algorithm_type="sde-dpmsolver++",
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(scheduler, device=self.device, sigmas=sampling_sigmas)
        elif sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = scheduler.timesteps
        else:
            raise ValueError(f"Unsupported solver: {sample_solver}")
        return scheduler, timesteps

    def _noise_level_to_step_idx(self, noise_level, timesteps):
        """Find the step index closest to a given noise level in [0, 1]."""
        target_t = noise_level * self.num_train_timesteps
        diffs = [(abs(float(t) - target_t), i) for i, t in enumerate(timesteps)]
        return min(diffs, key=lambda x: x[0])[1]

    def _denoise_to_clean(
        self,
        latent,
        scheduler,
        timesteps,
        start_idx,
        arg_c,
        arg_null,
        guide_scale,
        generator,
    ):
        """Denoise a single latent from start_idx to clean (step 0)."""
        scheduler._step_index = None
        scheduler.lower_order_nums = 0
        scheduler.model_outputs = [None] * scheduler.config.solver_order

        current = latent.squeeze(0) if latent.dim() == 5 else latent
        for t in timesteps[start_idx:]:
            latent_input = [current]
            t_tensor = torch.stack([t]).to(self.device)
            noise_pred_cond = self.model(latent_input, t=t_tensor, **arg_c)[0]
            noise_pred_uncond = self.model(latent_input, t=t_tensor, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            current = scheduler.step(
                noise_pred.unsqueeze(0), t, current.unsqueeze(0),
                return_dict=False, generator=generator,
            )[0].squeeze(0)
        return current

    def generate(
        self,
        input_prompt,
        size=(832, 480),
        frame_num=33,
        shift=5.0,
        sample_solver="dpm++",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        checkpoints=(1.0, 0.6, 0.3),
        transition_alpha=1,
        initial_population=6,
        valid_candidates=4,
        editing_ratio=0.5,
        eta_sde=0.5,
        verifier=None,
        mllm_model_name=None,
        mllm_base_url=None,
    ):
        """Run the full GEARS pipeline.

        Args:
            input_prompt: Text prompt.
            size: (W, H) output resolution.
            frame_num: Number of video frames.
            shift: Flow-matching shift factor.
            sample_solver: 'dpm++' or 'unipc'.
            sampling_steps: Total denoising steps.
            guide_scale: CFG scale.
            n_prompt: Negative prompt.
            seed: Random seed.
            offload_model: CPU offload between stages.
            checkpoints: Noise-level sequence (s_0, s_1, ..., s_k).
            transition_alpha: Stage boundary index (1-indexed over s_1..s_k).
            initial_population: N_0 initial candidates.
            valid_candidates: K_i candidates retained by the semantic gate.
            editing_ratio: q_edit fraction sent to Recycler.
            eta_sde: MA-SDE stochastic coefficient.
            verifier: VideoVLMRewardInference for multi-dim scoring.
            mllm_model_name: MLLM identifier passed to the API.
            mllm_base_url: OpenAI-compatible API endpoint.
        """
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if verifier is None:
            raise ValueError("verifier is required.")
        if not mllm_model_name or not mllm_base_url:
            raise ValueError("mllm_model_name and mllm_base_url are required.")

        target_shape = self._compute_target_shape(size, frame_num)
        seq_len = self._compute_seq_len(target_shape)
        def _encode_text(prompt_text, encode_device):
            return self.text_encoder([prompt_text], encode_device)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context_orig = _encode_text(input_prompt, self.device)
            context_null = _encode_text(n_prompt, self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context_orig = _encode_text(input_prompt, torch.device("cpu"))
            context_null = _encode_text(n_prompt, torch.device("cpu"))
            context_orig = [t.to(self.device) for t in context_orig]
            context_null = [t.to(self.device) for t in context_null]

        arg_null = {"context": context_null, "seq_len": seq_len}

        @contextmanager
        def noop_no_sync():
            yield
        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        initial_latents = torch.randn(
            (initial_population,) + tuple(target_shape),
            generator=seed_g, dtype=torch.float32, device=self.device,
        )

        candidate_pool = [
            {"latent": initial_latents[j], "prompt": input_prompt}
            for j in range(initial_population)
        ]

        carried_elites = []
        intermediate_videos = []

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            self.model.to(self.device)

            for ckpt_idx in range(1, len(checkpoints)):
                start_noise = checkpoints[ckpt_idx - 1]

                print(f"\n[GEARS] === Checkpoint {ckpt_idx}: denoise from "
                      f"s_{ckpt_idx-1}={start_noise:.2f} to 0 ===")
                print(f"[GEARS] Pool: {len(candidate_pool)} new + "
                      f"{len(carried_elites)} carried elites")

                scheduler, timesteps = self._build_scheduler(sample_solver, sampling_steps, shift)
                start_step_idx = self._noise_level_to_step_idx(start_noise, timesteps)

                clean_latents = []
                videos = []
                prompts = []
                for cand in tqdm(candidate_pool, desc=f"  Denoise ckpt {ckpt_idx}"):
                    if cand["prompt"] == input_prompt:
                        arg_c = {"context": context_orig, "seq_len": seq_len}
                    else:
                        if not self.t5_cpu:
                            self.text_encoder.model.to(self.device)
                            ctx = _encode_text(cand["prompt"], self.device)
                            if offload_model:
                                self.text_encoder.model.cpu()
                        else:
                            ctx = _encode_text(cand["prompt"], torch.device("cpu"))
                            ctx = [t.to(self.device) for t in ctx]
                        arg_c = {"context": ctx, "seq_len": seq_len}

                    clean = self._denoise_to_clean(
                        cand["latent"], scheduler, timesteps, start_step_idx,
                        arg_c, arg_null, guide_scale, seed_g,
                    )
                    clean_latents.append(clean)
                    prompts.append(cand["prompt"])

                if offload_model:
                    self.model.cpu()
                    torch.cuda.empty_cache()

                for cl in clean_latents:
                    videos.append(self.vae.decode([cl])[0])

                if offload_model:
                    self.model.to(self.device)

                reward_dicts = []
                for vid in videos:
                    reward_dicts.append(
                        get_multidim_reward(verifier, vid, input_prompt)
                    )

                for elite in carried_elites:
                    clean_latents.append(elite["clean_latent"])
                    videos.append(elite["video"])
                    prompts.append(elite["prompt"])
                    reward_dicts.append(elite["reward_dict"])

                total_scores = [rd["VQ"] + rd["MQ"] + rd["TA"] for rd in reward_dicts]
                print(f"[GEARS] Scores: {['%.3f' % s for s in total_scores]}")

                for j, vid in enumerate(videos):
                    intermediate_videos.append({
                        "stage": f"checkpoint_{ckpt_idx}",
                        "label": f"ckpt{ckpt_idx}_cand{j}",
                        "reward": total_scores[j],
                        "video": vid,
                    })

                routing = route_candidates(
                    reward_dicts=reward_dicts,
                    checkpoint_idx=ckpt_idx,
                    transition_alpha=transition_alpha,
                    keep_size=min(valid_candidates, len(reward_dicts)),
                    editing_ratio=editing_ratio,
                )

                stage_focus = get_stage_focus(ckpt_idx, transition_alpha)
                print(f"[GEARS] StageAdapter: focus={stage_focus}, "
                      f"elites={len(routing.elite_indices)}, "
                      f"recycle={len(routing.recycle_indices)}, "
                      f"discard={len(routing.discard_indices)}")

                is_last_checkpoint = (ckpt_idx == len(checkpoints) - 1)
                recycle_noise_level = checkpoints[ckpt_idx]
                recycle_start_idx = self._noise_level_to_step_idx(
                    recycle_noise_level,
                    timesteps,
                )
                recycled_survivors = []

                for idx in routing.recycle_indices:
                    print(f"[GEARS] Recycler candidate {idx} "
                          f"(score={total_scores[idx]:.3f}, focus={stage_focus})")

                    edit_scheduler, edit_timesteps = self._build_scheduler(
                        sample_solver, sampling_steps, shift
                    )
                    repaired_latent, enhanced_prompt = recycle_candidate(
                        model=self.model,
                        text_encoder=self.text_encoder,
                        clean_latent=clean_latents[idx],
                        video_tensor=videos[idx],
                        current_prompt=prompts[idx],
                        reward_dict=reward_dicts[idx],
                        checkpoint_noise_level=recycle_noise_level,
                        checkpoint_idx=ckpt_idx,
                        transition_alpha=transition_alpha,
                        scheduler=edit_scheduler,
                        timesteps=edit_timesteps,
                        start_step_idx=recycle_start_idx,
                        arg_null=arg_null,
                        guide_scale=guide_scale,
                        eta_sde=eta_sde,
                        num_train_timesteps=self.num_train_timesteps,
                        device=self.device,
                        mllm_model_name=mllm_model_name,
                        mllm_base_url=mllm_base_url,
                        generator=seed_g,
                        t5_cpu=self.t5_cpu,
                        offload_model=offload_model,
                    )
                    recycled_survivors.append({
                        "clean_latent": repaired_latent,
                        "prompt": enhanced_prompt,
                    })

                carried_elites = []
                for idx in routing.elite_indices:
                    carried_elites.append({
                        "clean_latent": clean_latents[idx],
                        "prompt": prompts[idx],
                        "reward_dict": reward_dicts[idx],
                        "total_score": total_scores[idx],
                        "video": videos[idx],
                    })

                if not is_last_checkpoint:
                    next_noise = checkpoints[ckpt_idx]
                    candidate_pool = []
                    for survivor in recycled_survivors:
                        renoised = renoise_latent(
                            survivor["clean_latent"],
                            next_noise,
                            seed_g,
                        )
                        candidate_pool.append({
                            "latent": renoised,
                            "prompt": survivor["prompt"],
                        })
                    print(f"[GEARS] Next pool: {len(candidate_pool)} re-noised + "
                          f"{len(carried_elites)} elites")
                else:
                    for survivor in recycled_survivors:
                        if offload_model:
                            self.model.cpu()
                            torch.cuda.empty_cache()
                        vid = self.vae.decode([survivor["clean_latent"]])[0]
                        if offload_model:
                            self.model.to(self.device)
                        rd = get_multidim_reward(verifier, vid, input_prompt)
                        carried_elites.append({
                            "clean_latent": survivor["clean_latent"],
                            "prompt": survivor["prompt"],
                            "reward_dict": rd,
                            "total_score": rd["VQ"] + rd["MQ"] + rd["TA"],
                            "video": vid,
                        })

        if not carried_elites:
            print("[GEARS] Warning: no surviving candidates.")
            return None

        best = max(carried_elites, key=lambda e: e["total_score"])
        print(f"\n[GEARS] Final best score: {best['total_score']:.4f}")

        self.intermediate_videos = intermediate_videos

        if offload_model:
            self.model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

        return best["video"]
