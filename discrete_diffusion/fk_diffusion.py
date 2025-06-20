import torch

import itertools

from reward_functions import (
    sentiment_score,
    toxicity_score,
    formality_score,
    gpt2_perp_score,
    cola_score,
    infinigram_perp_score,
    logmeanexp,
)
from mdlm.diffusion import Diffusion, _sample_categorical
from fkd_class import FKD

from tqdm import tqdm


def batch_inputs(inputs, batch_size):
    '''Batch inputs'''
    assert isinstance(inputs, list), 'inputs should be a list'

    batches = []
    for i in range(0, len(inputs), batch_size):
        batch = [inputs[j] for j in range(i, min(i + batch_size, len(inputs)))]
        batches.append(batch)

    return batches


def batched_infer(*, inputs, fn, batch_size):
    '''Use function on batched inputs'''
    batched_x = batch_inputs(inputs, batch_size)
    results = []
    for batch in batched_x:
        results.extend(fn(x_batch=batch))
    return results


def compute_rewards(*, samples, reward_name, reward_label):
    '''Compute log rewards'''
    if reward_name == 'sentiment':
        scores, _ = sentiment_score(texts=samples, label=reward_label)
    elif reward_name == 'toxicity':
        scores, _ = toxicity_score(
            texts=samples,
            label=reward_label,
            max_length=50,
        )  # for evals
    elif reward_name == 'formality':
        scores, _ = formality_score(texts=samples, label=reward_label)
    elif reward_name == 'gpt2_perp':
        scores, _ = gpt2_perp_score(texts=samples)
    elif reward_name == 'cola':
        scores, _ = cola_score(texts=samples, max_length=50)  # for evals
    elif reward_name.startswith('infinigram_perp_score'):
        max_num_samples = int(reward_name.split('-')[-1])
        max_ngram = int(reward_name.split('-')[-2])
        scores, _ = infinigram_perp_score(  # requires infinigram installed
            texts=samples, max_ngram=max_ngram, max_num_samples=max_num_samples
        )
    else:
        raise ValueError(f'Unknown reward function: {reward_name}')

    return scores


class FKDiffusion(Diffusion):
    '''
    FK Steering on Diffusion Model from MDLM
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _validate_configuration(self):
        assert self.config.loader.eval_batch_size == 1
        return super()._validate_configuration()

    def _ddpm_caching_update(self, x, t, dt, p_x0=None, n_x0_samples=4):
        raise NotImplementedError('Caching update not implemented')

    def _ddpm_update(self, x, t, dt, n_x0_samples=4):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.

        ### Added
        p_x0 = log_p_x0.exp()
        _x0_samples = [_sample_categorical(p_x0) for _ in range(n_x0_samples)]

        q_xs = p_x0 * (move_chance_t - move_chance_s)
        ###

        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)

        _x0_samples_with_copy = [
            copy_flag * x + (1 - copy_flag) * _x0 for _x0 in _x0_samples
        ]

        zs = copy_flag * x + (1 - copy_flag) * _x

        return zs, [x0_i.detach().cpu() for x0_i in _x0_samples_with_copy]

    def q_proposal_fn(self, x_batch, t, dt, num_x0_samples):
        z = [x['z'] for x in x_batch]
        z = torch.cat(z, dim=0)
        t = t * torch.ones(z.shape[0], 1, device=z.device)

        # TODO: implement caching
        new_z, sample = self._ddpm_update(z, t, dt, n_x0_samples=num_x0_samples)
        new_z = [z_i.unsqueeze(0) for z_i in new_z]

        # combine
        combined = []
        for i in range(len(new_z)):
            combined.append(
                {
                    'z': new_z[i],
                    'sample': [sample[j][i] for j in range(num_x0_samples)],
                }
            )

        return combined

    def prior_fn(self, batch_size_per_gpu, prompt_ids=None):
        z = self._sample_prior(batch_size_per_gpu, self.config.model.length).to(
            self.device
        )

        if prompt_ids is not None:
            z[:, : prompt_ids.shape[1]] = prompt_ids

        return {
            'z': z,
            'sample': [z] * self.config.fk_steering.num_x0_samples,
        }  # using z as sample for timestep 0, rather than predictions for x0

    def r_fn(self, x_batch, t, length_for_reward_fn):
        flatten_samples = []
        for x in x_batch:
            flatten_samples.extend(x['sample'])

        samples = torch.stack(flatten_samples, dim=0).squeeze(1)
        samples = samples[:, :length_for_reward_fn]
        samples = self.tokenizer.batch_decode(samples)

        scores = compute_rewards(
            samples=samples,
            reward_name=self.config.fk_steering.reward_fn,
            reward_label=self.config.fk_steering.reward_label,
        )

        # reshape and average
        scores = torch.tensor(scores)
        scores = scores.reshape(len(x_batch), self.config.fk_steering.num_x0_samples)
        # scores = scores.mean(dim=1).tolist()
        scores = logmeanexp(scores).tolist()
        return scores

    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5, prompt_text=None):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        assert batch_size_per_gpu == 1

        if num_steps is None:
            num_steps = self.config.sampling.steps

        length_for_reward_fn = self.config.fk_steering.reward_trim_length
        if prompt_text is not None:
            assert isinstance(prompt_text, str)
            prompt = self.tokenizer([prompt_text], return_tensors='pt', padding=False)
            prompt_ids = prompt['input_ids'][:, :-1].to(self.device)
            prompt_length = prompt_ids.shape[1]
            print(prompt, prompt_length, self.tokenizer.batch_decode(prompt_ids))
            length_for_reward_fn = (
                self.config.fk_steering.reward_trim_length + prompt_length
            )

        reward_batch_size = 8
        proposal_batch_size = 8
        dt = (1 - eps) / num_steps
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)

        fkd = FKD(
            potential_type=self.config.fk_steering.potential_type,
            lmbda=self.config.fk_steering.lmbda,
            num_particles=self.config.fk_steering.k_particles,
            adaptive_resampling=False,
            adaptive_resampling_at_end=False,
            resample_frequency=self.config.fk_steering.resample_frequency,
            resampling_t_start=-1,
            resampling_t_end=num_steps + 1,
            time_steps=num_steps + 1,
            reward_fn=lambda x: batched_infer(
                inputs=x,
                fn=lambda x_batch: self.r_fn(
                    x_batch=x_batch, t=None, length_for_reward_fn=length_for_reward_fn
                ),
                batch_size=reward_batch_size,
            ),
            device=self.device,
        )

        states = [
            self.prior_fn(batch_size_per_gpu, prompt_ids=prompt_ids)
            for _ in tqdm(list(range(self.config.fk_steering.k_particles)))
        ]

        rs_historic_means = [fkd.population_rs.detach().cpu().mean().item()]

        for i in tqdm(list(range(len(timesteps)))):
            assert not fkd._reached_terminal_sample
            t = timesteps[i]
            states_candidates = batched_infer(
                inputs=states,
                fn=lambda x_batch: self.q_proposal_fn(
                    x_batch, t, dt, self.config.fk_steering.num_x0_samples
                ),
                batch_size=proposal_batch_size,
            )
            states, _ = fkd.resample(
                sampling_idx=i, latents=states_candidates, x0_preds=states_candidates
            )
            rs_states = fkd.population_rs.detach().cpu()
            rs_historic_means.append(torch.mean(rs_states).item())

        assert fkd._reached_terminal_sample
        best_idx = torch.argmax(rs_states)
        best_sample = states[best_idx]
        best_r = rs_states[best_idx]

        return {
            'best': best_sample['z'],
            'best_r': best_r,
            'all_samples': states,
            'all_r': rs_states,
            'historic_means': rs_historic_means,
        }

    def restore_model_and_sample(self, num_steps, eps=1e-5, prompt_text=None):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()

        # Minor FK Modification: option to include prefilled prompt text

        results = self._sample(num_steps=num_steps, eps=eps, prompt_text=prompt_text)

        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.train()
        self.noise.train()
        return results
