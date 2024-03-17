import os
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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

class ActionPredictor(L.LightningModule):
    def __init__(self, action_space, device):
        super(ActionPredictor, self).__init__()
        self.save_hyperparameters()

        self.n_classes = action_space.n
        self.n_filters = 9 * 9 * 64
        self.backbone = nn.Sequential(nn.Conv2d(1, 32, stride=4, kernel_size=8))
        self.icnn = ICNN(32, 64, kernel_size=3, stride=1, padding=1, out_size=20, device=device)
        self.cnn = nn.Sequential(
           nn.Conv2d(64, 64, kernel_size=3, stride=2),
           nn.LeakyReLU(),)
        self.classifier = nn.Sequential( 
           nn.Flatten(),
           nn.LeakyReLU(),
           nn.Linear(64*9*9, 512),
           nn.Linear(512, self.n_classes)
        )
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=int(self.n_classes))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    

    def forward(self, states : torch.Tensor, next_states: torch.Tensor):
        x = torch.concat([states, next_states], dim=0)
        x = x.view(-1, 1, 84, 84) / 255.
        x = self.backbone(x)
        x1, x2, loss_1, loss_2 = self.icnn(x)
        x = self.cnn(x2)
        x = x.view(2, -1, 64, 9, 9).sum(dim=0)
        x = self.classifier(x)
        return x, loss_1, loss_2
    
    

    # ============================================
    # ============== STEP FUNCTIONS ==============
    # ============================================
    def step(self, batch, batch_idx):
        states, actions, _,_, next_states = batch
        x, loss_1, loss_2 = self.forward(states, next_states)

        one_hot_actions = F.one_hot(actions.squeeze(), self.n_classes).to(torch.float32)
        loss = F.mse_loss(x, one_hot_actions)
        loss += loss_1.mean()
        loss += loss_2.mean()

        self.valid_acc(x, one_hot_actions)

        return loss

    def training_step(self, batch, batch_idx):
        
        loss = self.step(batch, batch_idx)
        
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss)

        self.log('train_acc', self.valid_acc, on_step=True, on_epoch=False)

        return loss


    def test_step(self, batch, batch_idx):
        
        loss = self.step(batch, batch_idx)
        
        self.log("test_loss", loss)

        return loss
    

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
EARLY_STOPPING_PATIENCE = 10
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
    env_fn = lambda: gym.wrappers.AtariPreprocessing(gym.make("ALE/Pong-v5"), frame_skip=1)

    train_loader = EpisodeDataset(env_fn, num_envs=1, max_steps=1000, episodes_per_epoch=2, skip_first=0, repeat_action=1, device=DEVICE)

    # Create a PyTorch Lightning trainer with the generation callback
    wandb_logger = WandbLogger(**WANDB_KWARGS)
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "action_predictor"),
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, every_n_epochs=5),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor='val_loss', mode='min', patience=EARLY_STOPPING_PATIENCE, check_on_train_epoch_end=False),
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
        model = ActionPredictor(env_fn().action_space, device=DEVICE)
        trainer.fit(model, train_loader, val_dataloaders=train_loader)
    
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    result = {"val": val_result}
    return model, result

train_action_predictor()
