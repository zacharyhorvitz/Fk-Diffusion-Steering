"""
Feynman-Kac Diffusion (FKD) steering mechanism implementation.
"""

import torch
from enum import Enum
import numpy as np
from typing import Callable, Optional, Tuple, Union, List

# import logging


def list_tensor_idx(
    tensor_or_list: Union[torch.Tensor, List], indices: torch.Tensor
) -> Union[torch.Tensor, List]:
    if isinstance(tensor_or_list, list):
        return [tensor_or_list[i] for i in indices]
    else:
        return tensor_or_list[indices]


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"

    BON = "bon"  # Returns sorted particles
    IS = "is"


class FKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        lmbda: Lambda hyperparameter controlling weight scaling.
        num_particles: Number of particles to maintain in the population.
        adaptive_resampling: Whether to perform adaptive resampling.
        resample_frequency: Frequency (in timesteps) to perform resampling.
        resampling_t_start: Timestep to start resampling.
        resampling_t_end: Timestep to stop resampling.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        adaptive_resample_at_end: Whether to perform adaptive resampling at the end of the process (default: False).
        device: Device on which computations will be performed (default: CUDA).
        **kwargs: Additional keyword arguments, unused.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.0,
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        adaptive_resample_at_end: bool = False,
        device: torch.device = torch.device('cuda'),
        **kwargs,
    ) -> None:
        # Initialize hyperparameters and functions

        # if kwargs:
        # logging.warning(f"FKD Steering - Unused arguments: {kwargs}")

        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.adaptive_resample_at_end = adaptive_resample_at_end
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps

        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn

        # Initialize device and population reward state
        self.device = device

        # initial rewards
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * reward_min_value
        )
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)
        self._last_idx_sampled = -1
        self._reached_terminal_sample = False

        # construct resampling interval
        self.resampling_interval = np.arange(
            self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
        )
        # ensure that the last timestep is included
        self.resampling_interval = np.append(
            self.resampling_interval, self.time_steps - 1
        )

    def compute_reward(
        self, x0_preds: Union[torch.Tensor, List]
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, List]]:
        # Decode latents to population samples/images and compute rewards
        population_samples = self.latent_to_decode_fn(x0_preds)
        rs_candidates = self.reward_fn(population_samples)

        if isinstance(rs_candidates, list):
            rs_candidates = torch.tensor(rs_candidates).to(self.device)

        return rs_candidates, population_samples

    def resample(
        self,
        *,
        sampling_idx: int,
        latents: Union[torch.Tensor, List],
        x0_preds: Union[torch.Tensor, List],
    ) -> Tuple[Union[torch.Tensor, List], Optional[Union[torch.Tensor, List]]]:
        """
        Perform resampling of particles if conditions are met.
        Should be invoked at each timestep in the reverse diffusion process.

        Args:
            sampling_idx: Current sampling index (timestep). Should INCREASE from 0 to time_steps - 1.
            latents: Current noisy latents.
            x0_preds: Predictions for x0 based on latents.

        Returns:
            A tuple containing resampled latents and resampled samples/images.
        """

        # sanity check for sampling_idx. Some models might decrement the sampling_idx
        if sampling_idx <= self._last_idx_sampled:
            raise ValueError(
                f"Sampling index {sampling_idx} must be greater than last sampled index {self._last_idx_sampled}. Are you decrementing instead?"
            )
        self._last_idx_sampled = sampling_idx

        at_terminal_sample = sampling_idx == self.time_steps - 1
        self._reached_terminal_sample = at_terminal_sample

        if sampling_idx not in self.resampling_interval:
            return latents, None

        # While we can perform IS/BON with Gt = 1.0, we can also just skip the resampling step.
        if (
            self.potential_type in [PotentialType.BON, PotentialType.IS]
            and not at_terminal_sample
        ):
            return latents, None

        # Compute rewards
        rs_candidates, population_samples = self.compute_reward(x0_preds)

        # Compute potentials
        if self.potential_type == PotentialType.MAX:
            rs_candidates = torch.max(rs_candidates, self.population_rs)
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.ADD:
            rs_candidates = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
        elif self.potential_type == PotentialType.RT:
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type in [PotentialType.BON, PotentialType.IS]:
            assert at_terminal_sample
            w = torch.exp(self.lmbda * rs_candidates)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")

        # If we are at the last timestep, compute corrected potentials if using the MAX, ADD, or RT potential
        if at_terminal_sample:
            if (
                self.potential_type == PotentialType.MAX
                or self.potential_type == PotentialType.ADD
                or self.potential_type == PotentialType.RT
            ):
                w = torch.exp(self.lmbda * rs_candidates) / self.product_of_potentials

        # If we are using the BON potential, we just sort the particles by their rewards and return them
        if self.potential_type == PotentialType.BON:
            assert at_terminal_sample
            indices = torch.argsort(rs_candidates, descending=True)
            return list_tensor_idx(latents, indices), list_tensor_idx(
                population_samples, indices
            )

        w = torch.clamp(w, 0, 1e10)
        w[torch.isnan(w)] = 0.0

        # if all 0, set w to 1
        if w.sum() == 0:
            w = torch.ones_like(w)

        # If we are using adaptive resampling, check if we need to resample
        if self.adaptive_resampling or (
            at_terminal_sample and self.adaptive_resample_at_end
        ):
            # compute effective sample size
            normalized_w = w / w.sum()
            ess = 1.0 / (normalized_w.pow(2).sum())

            if ess < 0.5 * self.num_particles:
                print(f"Resampling at timestep {sampling_idx} with ESS: {ess}")
                # Resample indices based on weights
                indices = torch.multinomial(
                    w, num_samples=self.num_particles, replacement=True
                )

                resampled_latents = list_tensor_idx(latents, indices)
                self.population_rs = rs_candidates[indices]

                # Resample population images
                resampled_samples = list_tensor_idx(population_samples, indices)

                # Update product of potentials; used for max and add potentials
                self.product_of_potentials = (
                    self.product_of_potentials[indices] * w[indices]
                )
            else:
                # No resampling
                resampled_samples = population_samples
                resampled_latents = latents
                self.population_rs = rs_candidates

        else:
            indices = torch.multinomial(
                w, num_samples=self.num_particles, replacement=True
            )

            resampled_latents = list_tensor_idx(latents, indices)
            self.population_rs = rs_candidates[indices]

            # Resample population images
            resampled_samples = list_tensor_idx(population_samples, indices)

            # Update product of potentials; used for max and add potentials
            self.product_of_potentials = (
                self.product_of_potentials[indices] * w[indices]
            )

        return resampled_latents, resampled_samples


if __name__ == "__main__":
    # Demonstration of FKD resampling step
    import matplotlib.pyplot as plt
    import random

    # set seed
    random.seed(0)

    # 1x1 pixel images
    num_particles = 8
    x0s = torch.rand(num_particles, 1, 1)

    # reward darker images
    reward_function = lambda x: -0.5 * x.sum(dim=(1, 2))

    plt.rc('text', usetex=True)
    # fig, axs = plt.subplots(2, num_particles)

    # row for each potential type
    fig, axs = plt.subplots(1 + len(PotentialType), num_particles, figsize=(10, 10))

    axs[0, 0].set_title('Initial')
    for k in range(num_particles):
        axs[0, k].axis('off')
        axs[0, k].imshow(x0s[k].detach().numpy(), cmap='gray', vmin=0, vmax=1)

    for i, potential_type in enumerate(PotentialType):
        # Define the FKD steering mechanism
        fkds = FKD(
            potential_type=potential_type,
            lmbda=10.0,
            num_particles=num_particles,
            adaptive_resampling=False,
            adaptive_resample_at_end=False,
            resample_frequency=1,
            resampling_t_start=-1,
            resampling_t_end=1,
            time_steps=1,
            reward_fn=lambda x: reward_function(x),
            device=torch.device('cpu'),
        )

        # Define the sampling index
        sampling_idx = 0

        # Perform one resampling step
        resampled_latents, resampled_samples = fkds.resample(
            sampling_idx=sampling_idx,
            latents=x0s,
            x0_preds=x0s,
        )
        axs[i + 1, 0].set_title(f'Resampled ({potential_type})')
        for k in range(num_particles):
            axs[i + 1, k].axis('off')
            axs[i + 1, k].imshow(
                resampled_samples[k].detach().numpy(), cmap='gray', vmin=0, vmax=1
            )

    out_path = 'resampled_examples.png'
    plt.savefig(out_path)
    print('Saved resampled examples to:', out_path)
