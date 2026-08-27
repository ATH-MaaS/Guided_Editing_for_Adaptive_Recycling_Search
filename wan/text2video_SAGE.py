import math

import torch

from wan_evosearch import WanT2VEvoSearch


def compute_sage_mutation_std(sigmas, step_index, eta=1.0):
    """
    Args:
        sigmas: The full sigma schedule tensor from the scheduler.
        step_index: Current step index in the schedule.
        eta: Noise scale factor (default 1.0).

    Returns:
        Scalar tensor — the mutation standard deviation.
    """
    sigma = sigmas[step_index].float()
    next_sigma = sigmas[step_index + 1].float()
    delta_t = sigma - next_sigma
    # Clamp sigma away from 1.0 for numerical stability (log(0) guard)
    clamped_sigma = torch.clamp(sigma, max=1 - 3e-3)
    variance = eta ** 2 * (-delta_t + torch.log((1 - next_sigma) / (1 - clamped_sigma)))
    return torch.sqrt(torch.clamp(variance, min=1e-8))


class WanT2VEvolSAGE(WanT2VEvoSearch):
    def evosearch(self, latents_total, generation_steps, input_prompt, n_prompt,
                  arg_c, arg_null, guide_scale, sampling_steps, shift, sample_solver,
                  evolution_schedule, population_size_schedule, elite_size, mutation_rate,
                  reward_fn, generator, offload_model):
        number_of_n = len(latents_total)
        current_step = evolution_schedule[generation_steps]
        schedulers, timesteps = self._build_schedulers(number_of_n, sample_solver, sampling_steps, shift)

        # Store sigma schedule from the first scheduler for SAGE mutation computation
        scheduler_sigmas = schedulers[0].sigmas

        generation_steps_id = generation_steps
        std_list = self._std_list

        for step_idx, t in zip(range(current_step, sampling_steps), timesteps[current_step:]):
            next_latents = []
            for k in range(number_of_n):
                next_latent, variance, std = self._denoise_step(
                    latents_total[k], t, schedulers[k], arg_c, arg_null,
                    guide_scale, generator)
                next_latents.append(next_latent)
                if step_idx in evolution_schedule:
                    self.population_list[generation_steps_id].append(next_latent)
                    if variance is not None:
                        self.variance_list[generation_steps_id].append(variance)
            latents_total = torch.cat(next_latents)
            if step_idx in evolution_schedule:
                std_list[generation_steps_id] = std
                generation_steps_id += 1

        # Decode all candidates and score (offload DiT to free VRAM for VAE)
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        decoded_videos = []
        for k in range(number_of_n):
            videos = self.vae.decode([latents_total[k]])
            video = videos[0]
            self.video_list.append(video)
            decoded_videos.append(video)
        rewards = reward_fn(decoded_videos, input_prompt).to(self.device)
        if offload_model:
            self.model.to(self.device)

        # Cross-generation reward accumulation
        for gen_id in range(generation_steps, len(self.rewards_list)):
            self.rewards_list[gen_id].append(rewards)

        # Merge accumulated population and rewards for current generation
        population = torch.cat(self.population_list[generation_steps])
        accumulated_rewards = torch.cat(self.rewards_list[generation_steps])

        # Elite selection based on accumulated rewards
        elite_rewards, elite_indices = torch.topk(accumulated_rewards, elite_size)
        if elite_rewards[0] > self.best_reward:
            self.best_reward = elite_rewards[0].item()
            best_global_idx = elite_indices[0].item()
            self.best_video = self.video_list[best_global_idx]

        elites = population[elite_indices]

        # Tournament selection + mutation
        next_pop_size = population_size_schedule[generation_steps + 1]
        num_children = next_pop_size - elite_size

        if num_children <= 0:
            new_population = elites[:next_pop_size]
        else:
            parents = []
            tournament_size = max(1, int(population.shape[0] * 0.9))
            for _ in range(num_children):
                candidates = torch.randperm(population.shape[0])[:tournament_size]
                candidate_rewards = accumulated_rewards[candidates]
                winner = candidates[torch.argmax(candidate_rewards)]
                parents.append(population[winner])
            parents = torch.stack(parents)

            if generation_steps == 0:
                children = parents * math.sqrt(1 - mutation_rate ** 2) + mutation_rate * torch.randn_like(parents)
            else:
                evolution_step_index = evolution_schedule[generation_steps]
                sage_std = compute_sage_mutation_std(scheduler_sigmas, evolution_step_index)
                children = parents + sage_std * torch.randn_like(parents)

            new_population = torch.cat([elites, children])
        return new_population

def latent_to_decode(vae, latent):
    return vae.decode(latent[None])
