from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from dataclasses import dataclass
import pytorch_lightning as pl
import torch.nn as nn
import torch
import wandb
import os
from einops import rearrange
import torchvision.utils as vutils

def run_one_image( x, model, mask_ratio=0.75):

    loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
    y = model.unpatchify(y).detach().cpu()
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask).detach().cpu()  
    x = x.detach().cpu()
    return x, y, mask

@dataclass
class OptimCfg:
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    
def denormalize(img, dataset_name = 'nako',  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if dataset_name != 'nako':

        mean = torch.tensor(mean).view(3,1,1)
        std = torch.tensor(std).view(3,1,1)
    else:

        mean = torch.tensor([0.736, 0.380, 0.131]).view(3,1,1)
        std = torch.tensor([0.0367, 0.0354, 0.021]).view(3,1,1)
    return img * std + mean

class MAELightning(pl.LightningModule):
   

    def __init__(
        self,
        mae_model: nn.Module,
        mask_ratio: float = 0.75,
        optim_cfg: OptimCfg = OptimCfg(),
        use_cosine_schedule: bool = True,
        # max_epochs: Optional[int] = None,
        compile_model: bool = False,
        val_dataset: torch.utils.data.Dataset = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["mae_model", "val_dataset"])  # keeps config in checkpoints

        self.mae = mae_model
        self.mask_ratio = mask_ratio
        self.optim_cfg = optim_cfg
        self.use_cosine_schedule = use_cosine_schedule
        self.val_dataset = val_dataset

        if compile_model and hasattr(torch, "compile"):
            try:
                self.mae = torch.compile(self.mae)  
            except Exception as e:
                print(f"torch.compile failed, continuing uncompiled: {e}")


    def forward(self, imgs: torch.Tensor):
        return self.mae(imgs, mask_ratio=self.mask_ratio)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        loss, pred, mask = self(imgs)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=imgs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        loss, pred, mask = self(imgs)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=imgs.size(0))

        N = 10
        if batch_idx == 0:  
            if (self.current_epoch % N == 0) or (self.current_epoch == self.trainer.max_epochs - 1):
                self._log_reconstruction(to_view=[0, 1, 2, 3], epoch=self.current_epoch)
    def test_step(self, batch, batch_idx):
        imgs, _ = batch
        loss, pred, mask = self(imgs)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, batch_size=imgs.size(0))

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.optim_cfg.lr,
                     betas=self.optim_cfg.betas,
                     weight_decay=self.optim_cfg.weight_decay)

        if not self.use_cosine_schedule:
            return opt

        sched = CosineAnnealingLR(opt, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


    @torch.no_grad()
    def _log_reconstruction(self, to_view, epoch, tag="mae_output"):
        val_img = torch.stack([self.val_dataset[i][0] for i in to_view]).to(self.device)
        print(f"\n \n val image min, max {val_img.min()} {val_img.max()}")
        val_img, pred, mask = run_one_image(val_img, self.mae, mask_ratio=self.hparams.mask_ratio)

        masked_img = val_img * (1 - mask)

        im_paste = val_img * (1 - mask) + pred * mask


        concat_img = torch.cat([val_img, masked_img, im_paste], dim=0)  
        grid = vutils.make_grid(concat_img, nrow=val_img.size(0), pad_value=0) 
        grid = grid.clone()
        grid = grid.clamp(0, 1)
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {f"{tag}/epoch_{epoch}": wandb.Image(grid, caption=f"epoch {epoch}")},
                step=self.global_step,
            
            )

        recon_dir = os.path.join(self.logger.save_dir, "reconstructions")
        os.makedirs(recon_dir, exist_ok=True)
        save_path = os.path.join(recon_dir, f"{tag}_epoch_{epoch}.png")
        vutils.save_image(grid, save_path)
        print(f"[Epoch {epoch}] Reconstruction saved to {save_path}")
    
    