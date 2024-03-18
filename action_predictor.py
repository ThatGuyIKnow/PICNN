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
from torchvision.models import resnet50


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

        self.n_classes = 6
        self.n_filters = 9 * 9 * 64


        self.base_model = nn.Sequential(*list(resnet50(pretrained=True).children())[:-3])
        M = 1024
        self.classifier = nn.Sequential(*list(resnet50(pretrained=True).children())[-3:-2],
                                        nn.Flatten(),
                                        nn.Linear(2048*3*3, self.n_classes),
                                        nn.Softmax()
                                        )
        self.icnn = ICNN(M, M, kernel_size=3, stride=1, padding=1, out_size=6, device=device)

        # self.backbone = nn.Sequential(nn.Conv2d(1, 32, kernel_size=8, stride=4),
                                                
        #                                         nn.LeakyReLU(),
        #                                         nn.Dropout(0.5),
        #                                         nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #                                         nn.LeakyReLU()
        #                                     )
        # self.classifier = nn.Sequential( 
        # #    nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #    nn.Flatten(),
        #    nn.Dropout(0.5),
        #    nn.Linear(self.n_filters*2, 512),
        #    nn.Dropout(0.5),
        #    nn.Linear(512, self.n_classes),
        #    nn.Softmax()
        # )
        
        # self.reconstruction = nn.Sequential(
        #     # Start with the last layer's features (64 channels) and work backwards.
        #     # Transposed convolutional layer to upsample from 64 to 32 channels.
        #     # The kernel size and stride should mirror those of the corresponding Conv2d layer in the backbone.
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
        #     nn.LeakyReLU(),
        #     # Dropout layer is usually not used in the reconstruction path, but you can experiment with it.
            
        #     # Final layer to upsample and reduce channels back to the original 2 channels of the input.
        #     # Match the kernel size and stride of the first Conv2d layer in the backbone.
        #     nn.ConvTranspose2d(32, 2, kernel_size=8, stride=4),
        #     # Optionally, add a final activation function depending on the input data range
        #     # e.g., nn.Sigmoid() if your input data is normalized to [0, 1]
        # )

        
        self.valid_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=int(self.n_classes), average='none')
        
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}
    

    def forward(self, states : torch.Tensor, next_states: torch.Tensor):
        states = states.squeeze(dim=1).movedim(-1, 1)
        next_states = next_states.squeeze(dim=1).movedim(-1, 1)
        z = self.base_model(states)
        
        z_next = self.base_model(next_states)

        x1, x2, loss_1, loss_2 = self.icnn(z)
        x1, x2_next, loss_1, loss_2 = self.icnn(z_next)
        x = x2 + x2_next

        x = self.classifier(x)
        return x, x2, x2_next, (loss_1 + loss_2).sum(), (loss_1 + loss_2).sum()
    
    

    # ============================================
    # ============== STEP FUNCTIONS ==============
    # ============================================
    def step(self, batch, batch_idx):
        states, actions, _,_, next_states = batch
        states = states / 255.
        next_states = next_states / 255.
        x, z, z_next, loss_1, loss_2 = self.forward(states, next_states)

        actions = actions.squeeze()
        one_hot_actions = F.one_hot(actions, self.n_classes).to(torch.float32)
        ce_loss = self.loss_fn(x, one_hot_actions)
        loss_1 = loss_1.sum()
        loss_2 = loss_2.sum()
        
        loss = ce_loss + loss_1 + loss_2

        acc = torchmetrics.functional.accuracy(x.argmax(dim=-1), actions, task='multiclass', num_classes=int(self.n_classes), average='none')

        return loss, ce_loss, loss_1, loss_2, acc

    def training_step(self, batch, batch_idx):
        
        loss, ce_loss, loss_1, loss_2, acc = self.step(batch, batch_idx)
        
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_ce_loss", ce_loss, on_step=True, on_epoch=False)
        self.log("train_local_loss_1", loss_1, on_step=True, on_epoch=False)
        self.log("train_local_loss_2", loss_2, on_step=True, on_epoch=False)
        return loss


    def validation_step(self, batch, batch_idx):
        
        loss, ce_loss, loss_1, loss_2, acc = self.step(batch, batch_idx)
        
        self.log("valid_loss", loss)
        self.log("valid_ce_loss", ce_loss)
        self.log("valid_local_loss_1", loss_1)
        self.log("valid_local_loss_2", loss_2)
        
        self.log_dict({f'valid_acc.{i}': a for i, a in enumerate(acc)})

        return loss


    def test_step(self, batch, batch_idx):
        
        loss, ce_loss, loss_1, loss_2, acc = self.step(batch, batch_idx)
        
        self.log("test_loss", loss)
        self.log("test_ce_loss", ce_loss)
        self.log("test_local_loss_1", loss_1)
        self.log("test_local_loss_2", loss_2)

        return loss
    

# load and wrap the environment

SEED = 43
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
    env_fn = lambda: gym.wrappers.AtariPreprocessing(gym.make("ALE/Pong-v5"), frame_skip=1, grayscale_obs=False)

    train_loader = EpisodeDataset(env_fn, num_envs=1, max_steps=300, episodes_per_epoch=10, skip_first=0, repeat_action=1, device=DEVICE)

    # Create a PyTorch Lightning trainer with the generation callback
    wandb_logger = WandbLogger(**WANDB_KWARGS)
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "action_predictor"),
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            # ModelCheckpoint(save_weights_only=True, every_n_epochs=5),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor='valid_loss', mode='min', patience=EARLY_STOPPING_PATIENCE, check_on_train_epoch_end=False),
        ],
        # logger=wandb_logger,
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
