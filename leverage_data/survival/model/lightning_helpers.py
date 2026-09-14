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

def run_one_image( x, model):

    loss, y, mask = model(x.float())
    y = model.unpatchify(y).detach().cpu()
    # y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask).detach().cpu()  # 1 is removing, 0 is keeping
    # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = x.detach().cpu()
    # x = torch.einsum('nchw->nhwc', x).detach().cpu()

    return x, y, mask

@dataclass
class OptimCfg:
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    
def denormalize(img, dataset_name = 'nako',  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # print('dataset name is', dataset_name)
    if dataset_name != 'nako':

        mean = torch.tensor(mean).view(3,1,1)
        std = torch.tensor(std).view(3,1,1)
    else:

        mean = torch.tensor([0.736, 0.380, 0.131]).view(3,1,1)
        std = torch.tensor([0.0367, 0.0354, 0.021]).view(3,1,1)
    return img * std + mean

class SwinMAELightning(pl.LightningModule):
    """
    A thin LightningModule around MaskedAutoencoderViT.

    Expects batches as either:
      - a tensor of shape (B, C, H, W), or
      - a dict with key 'image' -> tensor (B, C, H, W).
    """

    def __init__(
        self,
        swin_model: nn.Module,
        patch_size: int = 4,
        optim_cfg: OptimCfg = OptimCfg(),
        use_cosine_schedule: bool = True,
        mode: str = '',
        # max_epochs: Optional[int] = None,
        compile_model: bool = False,
        val_dataset: torch.utils.data.Dataset = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["swin_model", "val_dataset"])  # keeps config in checkpoints

        self.swin = swin_model
        self.mode = mode
        self.patch_size = patch_size
        self.optim_cfg = optim_cfg
        self.use_cosine_schedule = use_cosine_schedule
        self.val_dataset = val_dataset

        if compile_model and hasattr(torch, "compile"):
            try:
                self.swin = torch.compile(self.swin)  # type: ignore[attr-defined]
            except Exception as e:
                print(f"torch.compile failed, continuing uncompiled: {e}")

    # -------------- utils --------------

    def forward(self, imgs: torch.Tensor, return_feats: bool = False):
        if return_feats and hasattr(self.swin, "forward_encoder"):
            latent, mask = self.swin.forward_encoder(imgs)
            return latent  # (B, N_visible, D)
        else:
            return self.swin(imgs)

    # -------------- steps --------------
    def training_step(self, batch, batch_idx):
        imgs, *_ = batch
        loss, _, _ = self(imgs)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=imgs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, *_ = batch
        loss, pred, mask = self(imgs)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=imgs.size(0))

        # log a small reconstruction preview for the first batch of the epoch
        N = 10
        if batch_idx == 0:  # log only once per epoch
            if (self.current_epoch % N == 0) or (self.current_epoch == self.trainer.max_epochs - 1):
                self._log_reconstruction(to_view=[0, 1, 2, 3], epoch=self.current_epoch)
    def test_step(self, batch, batch_idx):
        imgs, *_ = batch
        # imgs = self._get_images(batch)
        loss, _, _ = self(imgs)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, batch_size=imgs.size(0))

    # -------------- optimizer/scheduler --------------
    def configure_optimizers(self):
        # typical MAE uses AdamW
        opt = AdamW(self.parameters(), lr=self.optim_cfg.lr, betas=self.optim_cfg.betas, weight_decay=self.optim_cfg.weight_decay)

        if not self.use_cosine_schedule:
            return opt

        # Cosine over max_epochs; Lightning calls scheduler each epoch by default
        # max_epochs = self.max_epochs_for_sched or self.trainer.max_epochs or 100
        sched = CosineAnnealingLR(opt, T_max = self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


    @torch.no_grad()
    def _log_reconstruction(self, to_view, epoch, tag="mae_output"):
        # prepare images
        val_img = torch.stack([self.val_dataset[i][0] for i in to_view]).to(self.device)
        print(f"\n \n val image min, max {val_img.min()} {val_img.max()}")
        val_img, pred, mask = run_one_image(val_img, self.swin)

        # masked image
        masked_img = val_img * (1 - mask)

        # reconstruction pasted with visible patches
        im_paste = val_img * (1 - mask) + pred * mask


        concat_img = torch.cat([val_img, masked_img, im_paste], dim=0)  # shape (N*3, C, H, W)

        # make a grid image [C, H_total, W_total]
        # img = rearrange(concat_img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
        grid = vutils.make_grid(concat_img, nrow=val_img.size(0), pad_value=0)  # shape [3, H_grid, W_grid]
        grid = grid.clone()
        grid = grid.clamp(0, 1)
        # grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-5)

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
    
class MAELightning(pl.LightningModule):
    """
    A thin LightningModule around MaskedAutoencoderViT.

    Expects batches as either:
      - a tensor of shape (B, C, H, W), or
      - a dict with key 'image' -> tensor (B, C, H, W).
    """

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
                self.mae = torch.compile(self.mae)  # type: ignore[attr-defined]
            except Exception as e:
                print(f"torch.compile failed, continuing uncompiled: {e}")

    # -------------- utils --------------

    def forward(self, imgs: torch.Tensor):
        return self.mae(imgs, mask_ratio=self.mask_ratio)

    # -------------- steps --------------
    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        loss, pred, mask = self(imgs)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=imgs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        loss, pred, mask = self(imgs)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=imgs.size(0))

        # log a small reconstruction preview for the first batch of the epoch
        N = 10
        if batch_idx == 0:  # log only once per epoch
            if (self.current_epoch % N == 0) or (self.current_epoch == self.trainer.max_epochs - 1):
                self._log_reconstruction(to_view=[0, 1, 2, 3], epoch=self.current_epoch)

    def test_step(self, batch, batch_idx):
        imgs, _ = batch
        # imgs = self._get_images(batch)
        loss, pred, mask = self(imgs)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, batch_size=imgs.size(0))

    # -------------- optimizer/scheduler --------------
    def configure_optimizers(self):
        # typical MAE uses AdamW
        opt = AdamW(self.parameters(), lr=self.optim_cfg.lr, betas=self.optim_cfg.betas, weight_decay=self.optim_cfg.weight_decay)

        if not self.use_cosine_schedule:
            return opt

        # Cosine over max_epochs; Lightning calls scheduler each epoch by default
        # max_epochs = self.max_epochs_for_sched or self.trainer.max_epochs or 100
        sched = CosineAnnealingLR(opt, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


    @torch.no_grad()
    def _log_reconstruction(self, to_view, epoch, tag="mae_output"):
        # prepare images
        val_img = torch.stack([self.val_dataset[i][0] for i in to_view]).to(self.device)
        print(f"\n \n val image min, max {val_img.min()} {val_img.max()}")
        val_img, pred, mask = run_one_image(val_img, self.mae, mask_ratio=self.hparams.mask_ratio)

        # masked image
        masked_img = val_img * (1 - mask)

        # reconstruction pasted with visible patches
        im_paste = val_img * (1 - mask) + pred * mask


        concat_img = torch.cat([val_img, masked_img, im_paste], dim=0)  # shape (N*3, C, H, W)

        # make a grid image [C, H_total, W_total]
        # img = rearrange(concat_img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
        grid = vutils.make_grid(concat_img, nrow=val_img.size(0), pad_value=0)  # shape [3, H_grid, W_grid]
        grid = grid.clone()
        grid = grid.clamp(0, 1)
        # grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-5)

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
    
    