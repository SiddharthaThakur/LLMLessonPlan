import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class LitLinear(pl.LightningModule):
    """
    Smallest possible LightningModule.
    Replaces the usual 'hello world' with a trainable y = Wx + b.
    """
    def __init__(self, in_dim: int = 10, out_dim: int = 1, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

    def _step(self, batch, stage: str):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
