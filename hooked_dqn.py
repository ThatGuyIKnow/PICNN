from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from typing import Any, Mapping, Optional, Tuple, Union

import copy
import math
import gym
import gymnasium

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


HOOKED_DEFAULT_CONFIG = {
    'pre_interaction_hook': None,
    'interaction_hook': None,
    'post_interaction_hook': None,
    'additional_loss_hook': None,
}

class HookedDQN(DQN):
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Custom DQN Agent with Hooks

        :param pre_interaction_hook: Function to call before interaction
        :param interaction_hook: Function to call during action selection
        :param post_interaction_hook: Function to call after interaction
        """
        _cfg = copy.deepcopy(HOOKED_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models, memory, observation_space, action_space, device, _cfg)
        self._pre_interaction_hook = _cfg["pre_interaction_hook"]
        self._interaction_hook = _cfg["interaction_hook"]
        self._post_interaction_hook = _cfg["post_interaction_hook"]
        self._additional_loss_hook = _cfg["additional_loss_hook"]

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Call the pre-interaction hook if it exists, then proceed with the base implementation."""
        if self._pre_interaction_hook is not None:
            self._pre_interaction_hook(timestep, timesteps)
        super().pre_interaction(timestep, timesteps)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Call the interaction hook if it exists, then proceed with the base action selection."""
        if self._interaction_hook is not None:
            self._interaction_hook(states, timestep, timesteps)
        return super().act(states, timestep, timesteps)



    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts and not timestep % self._update_interval:
            self._update(timestep, timesteps)

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

        if self._post_interaction_hook is not None:
            self._post_interaction_hook(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
            self.memory.sample(names=self.tensors_names, batch_size=self._batch_size)[0]

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            sampled_states = self._state_preprocessor(sampled_states, train=True)
            sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

            # compute target values
            with torch.no_grad():
                next_q_values, _, _ = self.target_q_network.act({"states": sampled_next_states}, role="target_q_network")

                target_q_values = torch.max(next_q_values, dim=-1, keepdim=True)[0]
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_q_values

            # compute Q-network loss
            
            q_values = torch.gather(self.q_network.act({"states": sampled_states}, role="q_network")[0],
                                    dim=1, index=sampled_actions.long())

            q_network_loss = F.mse_loss(q_values, target_values)

            # optimize Q-network
            self.optimizer.zero_grad()
            q_network_loss.backward()
            self.optimizer.step()

            if self._additional_loss_hook is not None:
                
                addition_loss = self._additional_loss_hook(self, Variable(sampled_states.data, requires_grad=True), sampled_actions, sampled_rewards, Variable(sampled_next_states.data, requires_grad=True), sampled_dones)
                self.optimizer.zero_grad()
                addition_loss.backward()
                self.optimizer.step()

            # update target network
            if not timestep % self._target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

            # record data
            self.track_data("Loss / Q-network loss", q_network_loss.item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
