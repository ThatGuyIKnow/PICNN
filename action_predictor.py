import os
from types import ModuleType
from typing import Sequence, Tuple
import gymnasium as gym

import torch
import torch.nn as nn
import torchmetrics

from EpisodeDataset import EpisodeDataset
from icnn import ICNN

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision.models import vgg16
from torchvision.transforms import Resize


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb

from model import LayerNormChannelLast, cnn_forward
from models import CNN

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
  import matplotlib.pyplot as plt
import torch
import numpy as np

def action_to_image(action, title="Action", figsize=(2, 2)):
    """
    Convert an action (scalar or vector) to an image representation.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(action)), action+0.1)
    ax.set_title(title)

    # Convert the Matplotlib plot to an image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

class LogCompositeStateActionCallback(L.Callback):
    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if (trainer.global_step + 1) % self.log_every_n_steps == 0:
            state, action, reward, done, next_state = batch
            predicted_actions = outputs.get('preds', torch.tensor([]))

            if trainer.logger and trainer.logger.experiment:
                for i in range(1):

                    resize_op = Resize(200)

                    state_img = resize_op(state[i].unsqueeze(0)).cpu().numpy()  # Assuming state is an image tensor
                    next_state_img = resize_op(next_state[i].unsqueeze(0)).cpu().numpy()  # Assuming state is an image tensor
                    action_img = action_to_image(action[i].cpu().numpy(), title="Action").transpose(2, 0, 1)
                    pred_action_img = action_to_image(predicted_actions[i].detach().cpu().numpy(), title="Predicted Action").transpose(2, 0, 1)

                    state_img = np.repeat(state_img, 3, axis=0)
                    next_state_img = np.repeat(next_state_img, 3, axis=0)
                    overlay_img = (state_img / 2) + (next_state_img / 2)

                    # Concatenate images
                    composite_img = np.hstack((
                        np.dstack((state_img, next_state_img, overlay_img)), 
                        np.dstack((action_img, pred_action_img, pred_action_img * 0))
                    ))

                    # Convert composite image to Tensor and add batch dimension
                    composite_img_tensor = torch.tensor(composite_img).unsqueeze(0).float() / 255

                    trainer.logger.experiment.add_images('composite/state_action_pred', composite_img_tensor, global_step=trainer.global_step)


class ActionPredictor(L.LightningModule):
    # def __init__(self, action_space, device):
    #     super(ActionPredictor, self).__init__()
    #     self.save_hyperparameters()

    #     self.n_classes = 6
    #     self.n_filters = 9 * 9 * 64

    #     self.base_model = nn.Sequential(*list(vgg16(pretrained=True).features.children())[:-1])
    #     M = self.base_model[-2].out_channels
    #     self.add_conv = nn.Conv2d(in_channels=M, out_channels=M,
    #                               kernel_size=3, stride=1, padding=1)
    #     self.encoder = nn.Sequential(*list(vgg16(num_classes=32).classifier.children()))
    #     # create templates for all filters
    #     self.out_size = 14

    #     self.icnn = ICNN(M, M, kernel_size=3, stride=1, padding=1, out_size=self.out_size, device=device)

    #     self.classifier = nn.Sequential(nn.Linear(2048*3*3, self.n_classes),
    #                                     nn.Softmax())

    def __init__(
        self,
        image_size: Tuple[int, int],
        channels_multiplier: int,
        layer_norm: bool = True,
        activation: ModuleType = nn.SiLU,
        stages: int = 1,
    ) -> None:
        super(ActionPredictor, self).__init__()
        self.save_hyperparameters()
        self.n_classes = 6
        self.input_dim = (2, *image_size)
        self.model = nn.Sequential(
            CNN(
                input_channels=self.input_dim[0],
                hidden_channels=(torch.tensor([2**i for i in range(stages)]) * channels_multiplier).tolist(),
                cnn_layer=lambda input_size, output_size, **layer_args: nn.Conv2d(input_size, output_size, **layer_args, groups=2),
                layer_args={"kernel_size": 4, "stride": 2, "padding": 1, "bias": not layer_norm},
                activation=activation,
                norm_layer=[LayerNormChannelLast for _ in range(stages)] if layer_norm else None,
                norm_args=(
                    [{"normalized_shape": (2**i) * channels_multiplier, "eps": 1e-3} for i in range(stages)]
                    if layer_norm
                    else None
                ),
            ),
            nn.Flatten(-3, -1),
        )

        with torch.no_grad():
            self.output_dim = self.model(torch.zeros(1, *self.input_dim)).shape[-1]

        self.model.append(nn.Sequential(
            nn.Linear(self.output_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.n_classes),
            nn.Softmax(),))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, obs: Sequence[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(obs, dim=-3)  # channels dimension
        x = cnn_forward(self.model, x, x.shape[-3:], (-1,))
        return x.squeeze(dim=1)


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
    

    # def forward_prong(self, x: torch.Tensor):
    #     x = x.squeeze(dim=1).movedim(-1, 1)

    #     x = self.base_model(x)

    #     _, x, loss_1, loss_2 = self.icnn(x)

    #     x = self.encoder(x)

    #     return x.flatten(dim=1), loss_1, loss_2



    # def forward(self, states : torch.Tensor, next_states: torch.Tensor):
    #     enc_states, loss_1, loss_2= self.forward_prong(states)
    #     enc_next_states, loss_3, loss_4 = self.forward_prong(next_states)

    #     x = torch.concat([enc_states, enc_next_states], dim=-1)

    #     x = self.classifier(x)
    #     return x, (loss_1 + loss_2 + loss_3 + loss_4).sum()
    
    

    # ============================================
    # ============== STEP FUNCTIONS ==============
    # ============================================
    def step(self, batch, batch_idx):
        states, actions, _,_, next_states = batch
        states = states / 255.
        next_states = next_states / 255.
        x = self.forward([states, next_states])

        ce_loss = self.loss_fn(x, actions)
        loss = ce_loss

        acc = torchmetrics.functional.accuracy(x.argmax(dim=-1), actions.argmax(dim=-1), task='multiclass', num_classes=int(self.n_classes), average='none')

        return loss, ce_loss, 0, acc, x

    def training_step(self, batch, batch_idx):
        
        loss, ce_loss, local_loss, acc, pred = self.step(batch, batch_idx)
        
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_ce_loss", ce_loss, on_step=True, on_epoch=False)
        self.log("train_local_loss", local_loss, on_step=True, on_epoch=False)
        return {'loss': loss, 'preds': pred}


    def validation_step(self, batch, batch_idx):
        
        loss, ce_loss, local_loss, acc, pred = self.step(batch, batch_idx)
        
        self.log("valid_loss", loss)
        self.log("valid_ce_loss", ce_loss)
        self.log("valid_local_loss_1", local_loss)
        
        self.log_dict({f'valid_acc.{i}': a for i, a in enumerate(acc)})

        return loss


    def test_step(self, batch, batch_idx):
        
        loss, ce_loss, local_loss, acc, pred = self.step(batch, batch_idx)
        
        self.log("test_loss", loss)
        self.log("test_ce_loss", ce_loss)
        self.log("test_local_loss", local_loss)
        return loss
    
class LogisticRegression(L.LightningModule):
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Sequential(
            torch.nn.Linear(n_inputs, 512),
            torch.nn.Linear(512, n_outputs)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    # make predictionsd
    def forward(self, x):
        y_pred = F.softmax(self.linear(x / 225.))
        return y_pred
    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
    
    def training_step(self, batch, batch_idx):
        states, actions, _,_, next_states = batch
        x = torch.stack([states, next_states], dim=1).flatten(start_dim=1)
        pred = self.forward(x)
        loss = self.loss_fn(pred, actions)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        
        return {'loss': loss, 'preds': pred}
# load and wrap the environment

SEED = 42
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Model parameters
IMG_CHANNELS = 1

# Training parameters
BATCH_SIZE = 256
NUM_WORKERS = 4
MAX_EPOCHS = 100
SAVE_EVERY_N_EPOCHS = 5
VAL_SPLIT = 0.1
GRADIENT_CLIPPING_VAL = 0.5
IMAGE_VIS_COUNT = 8
EARLY_STOPPING_PATIENCE = 20
WANDB_KWARGS = {
    'log_model': "all", 
    'prefix': 'action_predictor', 
    'project': 'action_predictor',
}
CHECKPOINT_PATH = 'runs'  # Path to the folder where the pretrained models are saved

# Setting the seed for reproducibility
L.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_action_predictor():
    env_fn = lambda: gym.wrappers.AtariPreprocessing(gym.make("ALE/Pong-v5"), frame_skip=1, screen_size=84)

    train_data = EpisodeDataset(env_fn, num_envs=4, max_steps=300, episodes_per_epoch=1000, skip_first=0, repeat_action=1, device=DEVICE)
    #val_data = EpisodeDataset(env_fn, num_envs=1, max_steps=100, episodes_per_epoch=10, skip_first=0, repeat_action=1, device=DEVICE)


    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    #val_loader = DataLoader(val_data, batch_size=64, shuffle=False)


    # Create a PyTorch Lightning trainer with the generation callback
    wandb_logger = WandbLogger(**WANDB_KWARGS)
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "action_predictor"),
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            # ModelCheckpoint(save_weights_only=True, every_n_epochs=5),
            # LogCompositeStateActionCallback(log_every_n_steps=100),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor='valid_loss', mode='min', patience=EARLY_STOPPING_PATIENCE, check_on_train_epoch_end=False),
        ],
        logger=wandb_logger,
        # gradient_clip_val=GRADIENT_CLIPPING_VAL
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    # Check if a pretrained model exists; if not, train a new one
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "action_predictor_best.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ActionPredictor.load_from_checkpoint(pretrained_filename)
    else:
        # model = ActionPredictor(image_size=(84, 84), channels_multiplier=12)
        model = LogisticRegression(84*84*2, 6)
        trainer.fit(model, train_loader)
    
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    result = {"val": val_result}
    return model, result

train_action_predictor()
