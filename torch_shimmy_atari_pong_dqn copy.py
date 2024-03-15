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



class QNetwork(DeterministicMixin, Model):

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.n_classes = action_space.n
        self.n_filters = 5184
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


    def compute(self, inputs, role):
        return self.forward(inputs["states"].view(-1, 4, 84, 84) / 255.)['pred_2'], {}
    
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
    
TOTAL_TIMESTEPS = int(5e6)

cfg = DQN_DEFAULT_CONFIG.copy()
cfg["learning_starts"] = 80000
cfg["learning_rate"] = 1e-4
cfg["polyak"] = 1.0
cfg["batch_size"] = 32
cfg["polyak"] = 1.0
cfg["exploration"]["initial_epsilon"] = 1.0
cfg["exploration"]["final_epsilon"] = 0.01
cfg["exploration"]["timesteps"] = int(TOTAL_TIMESTEPS * 0.1)
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/ALE_Pong"
# cfg["experiment"]["wandb"] = True
# cfg["experiment"]["wandb_kwargs"] = {'project': 'inv_dyn'}

agent = DQN(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": TOTAL_TIMESTEPS, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()

trainer.eval()