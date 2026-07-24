import math
from dataclasses import dataclass
from typing import Optional, Any
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.strategies import DDPStrategy
from time_distance.src.load_data import load_areds, load_nako, load_new_nako
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from time_distance.MAE import  models_mae
from lightning_helpers import OptimCfg, MAELightning
from pytorch_lightning.loggers import WandbLogger
import datetime
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms

torch.set_float32_matmul_precision('medium')

ENTITY = 'success_vera'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--testrun", type=int, default=0)
    parser.add_argument("--ssl_method", type=str, default='')
    parser.add_argument("--dataset_name", type=str, default='nako')
    parser.add_argument("--comment", type=str, default='')
    parser.add_argument("--group_name", type=str, default='TMI')
    parser.add_argument("--project", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument("--compile", action="store_true")
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    start_time_fmt = start_time.strftime("%Y-%m-%d %H:%M:%S")

    args.testrun = bool(args.testrun)
    limit_train_batches = None
    limit_val_batches = None    
    limit_test_batches = None
    if args.testrun:
        limit_train_batches = 3
        limit_val_batches = 3
        limit_test_batches = 3
        args.epochs = 2


    train_trans = test_trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC), 
                                         transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0)),  
                                        transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomGrayscale(p=0.2),
                                        transforms.Normalize(mean=[0.419, 0.209, 0.122],
                                                             std = [0.280, 0.164, 0.113] )
                                        # transforms.Normalize(mean = [0.417, 0.201, 0.114],
                                        #                      std =  [0.265, 0.140, 0.090])
                                         
                                         ])

    dataset_train, dataset_test, dataset_val, transforms_ = load_nako( 
                        image_size = args.img_size, 
                        batch_size = args.batch_size, 
                        ssl_method = args.ssl_method,
                        return_loader=False,
                        normalize= True,
                        return_all = False,
                        augment_test=test_trans,
                        augment_train=train_trans
                                    )
    n_data2 = load_new_nako(image_size = args.img_size,  normalize = True, 
                            augment_train = train_trans, 
                         )
    combined_dataset = ConcatDataset([dataset_train, dataset_test, dataset_val, n_data2])
    train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    
    # train_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    config = vars(args)
    run_name = f'{args.ssl_method}_{transforms_}_imgsize_{args.img_size}_{args.epochs}_{args.batch_size}_{start_time_fmt}'

    exp_dir = f"../../results/TMI/{args.dataset_name}/{args.ssl_method}/arc_{args.model}_{args.comment}_{transforms_}_{args.epochs}_{args.batch_size}_MR_{args.mask_ratio}_PS_{args.patch_size}_{start_time_fmt}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)


    print("CUDA device count:", torch.cuda.device_count())
    #--------------------------------------------Model------------------------------------#

    mae = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss )

    # mae = MAELightning()

    optim_cfg = OptimCfg(lr=args.lr, weight_decay=args.weight_decay)
    lit = MAELightning(mae, compile_model=args.compile, optim_cfg = optim_cfg,
                        val_dataset=dataset_val, **config)
    wandb_logger = WandbLogger(
    project=args.project,   
    name=run_name,          
    save_dir=exp_dir,       
    entity = ENTITY,
    config = config,
    save_code=True, 
    group = args.group_name,
    resume="allow",  

)
    periodic_ckpt = ModelCheckpoint(
    dirpath=f"{exp_dir}/checkpoints",           
    filename="mae-{epoch}",           
    save_top_k=-1,                    
    every_n_epochs=10                 
)
    last_ckpt = ModelCheckpoint(
    dirpath=f"{exp_dir}/checkpoints",
    filename="mae-last",
    save_last=True             
)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        accelerator="gpu",
        devices = "auto",
        log_every_n_steps=25,
        strategy="ddp",   # 
        logger=wandb_logger,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        
        callbacks=[periodic_ckpt, last_ckpt]
    )

    trainer.fit(lit, train_loader, val_loader)
# ------------------- Example usage script -------------------
if __name__ == "__main__":
    main()
    
# python3 mae_lightning.py --batch_size 512 --dataset_name nako --ssl_method MAE --patch_size 14 --mask_ratio 0.5 --epochs 300 --testrun 1