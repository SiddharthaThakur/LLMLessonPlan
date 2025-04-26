import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models.lit_module import LitLinear
from data.synth_linear import SynthLinear

import hydra, os
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_ds = SynthLinear(n=cfg.data.n, in_dim=cfg.model.in_dim)
    val_ds   = SynthLinear(n=cfg.data.n_val, in_dim=cfg.model.in_dim)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size)

    model = LitLinear(**cfg.model)

    trainer = pl.Trainer(
        max_epochs   = cfg.train.epochs,
        accelerator  = "auto",
        devices      = 1 if torch.cuda.is_available() else None,
        log_every_n_steps = 10,
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
