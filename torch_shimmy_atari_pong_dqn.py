import gymnasium as gym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

class ClassSpecificDQN(DQN):
    
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super()._update(timestep, timesteps)
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


class QNetwork_(DeterministicMixin, Model):

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.n_classes = 3
        self.n_filters = 7 * 7 * 64
        self.alpha = 1
        self.beta = -1

        self.mu = torch.Tensor([self.beta + self.alpha * i for i in range(self.n_classes)])
        self.var = torch.tensor(1)

        self.backbone = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                                nn.LeakyReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.LeakyReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.LeakyReLU(),
                                                nn.Flatten(),
                                            )
        self.classifier = nn.Sequential(
            nn.Linear(self.n_filters, 512),
            nn.Linear(512, self.num_actions)
        )
        


        self.correlation = nn.Parameter(F.softmax(torch.rand(size=(self.n_classes, self.n_filters)), dim=0))

    def normal_distribution(self, x, mu, var):
        # Convert mu and var to tensors if they are not already, ensuring compatibility with x
        mu, var = torch.tensor(mu).to(x.device), torch.tensor(var).to(x.device)

        # Calculate the probability density
        coeff = 1 / torch.sqrt(2 * torch.pi * var)
        exponent = torch.exp(-(x - mu) ** 2 / (2 * var))
        return coeff * exponent
    
    def class_extractor(self, pred: torch.Tensor):

        x = pred.sum(dim=-1, keepdim=True) * (1/self.num_actions)

        return self.normal_distribution(x, self.mu, self.var.expand_as(x))



    def forward(self, inputs : torch.Tensor, targets=None, forward_pass='default'):
        features = self.backbone(inputs)
        pred_1 = self.classifier.forward(features)
        
        pred = self.class_extractor(pred_1) # prediction of discrimination pathway (using all filters)

        # sample class ID using reparameter trick
        pred_softmax = torch.softmax(pred, dim=-1)
        with torch.no_grad():
            sample_cat = torch.multinomial(pred_softmax, 1, replacement=False).flatten().to(device)
            ind_positive_sample = sample_cat == targets  # mark wrong sample results
            sample_cat_oh = F.one_hot(sample_cat, num_classes=pred.shape[1]).float().to(device)
            epsilon = torch.where(sample_cat_oh != 0, 1 - pred_softmax, -pred_softmax).detach()
        sample_cat_oh = pred_softmax + epsilon

        # sample filter using reparameter trick
        correlation_softmax = F.softmax(self.correlation, dim=0)
        correlation_samples = sample_cat_oh @ correlation_softmax
        with torch.no_grad():
            ind_sample = torch.bernoulli(correlation_samples).bool()
            epsilon = torch.where(ind_sample, 1 - correlation_samples, -correlation_samples)
        binary_mask = correlation_samples + epsilon
        feature_mask = features * binary_mask  # binary
        pred_2 = self.classifier(feature_mask) # prediction of Interpretation pathway (using a cluster of class-specific filters)
        # with torch.no_grad(): 
        #     correlation_samples = correlation_softmax[targets]
        #     binary_mask = torch.bernoulli(correlation_samples).bool()
        #     feature_mask_self = features * ~binary_mask
        #     pred_3 = self.classifier(feature_mask_self) # prediction of Interpretation pathway (using complementary clusters of class-specific filters)
        out = {"features": features, 'pred_1': pred_1, 'pred_2': pred_2, #'pred_3': pred_3,
                    'ind_positive_sample': ind_positive_sample}
        return out


    def compute(self, inputs, role):
        return self.forward(inputs["states"].view(-1, 4, 84, 84) / 255.)['pred_2'], {}
    
class QNetwork(DeterministicMixin, Model):

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.n_classes = 3
        self.n_filters = 7 * 7 * 64
        self.alpha = 1
        self.beta = -1

        self.backbone = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                                nn.LeakyReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.LeakyReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.LeakyReLU(),
                                                nn.Flatten(),
                                                nn.Linear(self.n_filters, 512),
                                                nn.Linear(512, self.num_actions)
                                            )

    def compute(self, inputs, role):
        return self.backbone(inputs["states"].view(-1, 4, 84, 84) / 255.), {}
    
# load and wrap the environment
env = gym.make("ALE/Pong-v5")
env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
env = gym.wrappers.FrameStack(env, 4)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=30000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators).
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#models
models = {}
models["q_network"] = QNetwork(env.observation_space, env.action_space, device)
models["target_q_network"] = QNetwork(env.observation_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#configuration-and-hyperparameters
cfg = DQN_DEFAULT_CONFIG.copy()
cfg["learning_starts"] = 100
cfg["exploration"]["initial_epsilon"] = 1.0
cfg["exploration"]["final_epsilon"] = 0.1
cfg["exploration"]["timesteps"] = 100000
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/ALE_Pong"
cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"] = {'project': 'inv_dyn'}

agent = DQN(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": int(1e6), "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()

trainer.eval()