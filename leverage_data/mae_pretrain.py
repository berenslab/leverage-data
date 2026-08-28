import yaml
import argparse
import torch
import torch.nn.functional as F
from data.load_data import load_nako, load_new_nako
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models import models_mae
from utils.lightning_helpers import OptimCfg, MAELightning
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
    parser.add_argument("--use_imagenet_weights", type=int, default=0)
    parser.add_argument("--blr", type=float, default=1.5e-4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--eff_batch_size", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument("--compile", action="store_true")
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--weights_name', default='mae_pretrain_vit_base', type=str, metavar='weights',
                        help='Name of model to train')
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    start_time_fmt = start_time.strftime("%Y-%m-%d %H:%M:%S")

    args.testrun = bool(args.testrun)
    args.use_imagenet_weights = bool(args.use_imagenet_weights)
    num_gpus_total = args.num_devices * args.num_nodes
    args.batch_size = args.eff_batch_size//num_gpus_total//args.accum_iter


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

    exp_dir = f"pre_train/{args.dataset_name}/{args.ssl_method}/arc_{args.model}_{args.comment}_{transforms_}_{args.epochs}_{args.batch_size}_MR_{args.mask_ratio}_PS_{args.patch_size}_{start_time_fmt}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)


    print("CUDA device count:", torch.cuda.device_count())
    #--------------------------------------------Model------------------------------------#
    mae_model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss )
    if args.use_imagenet_weights:
        
        imagenet_checkpoint = torch.load(os.path.join('weights', f"{args.weights_name}.pth"), map_location = 'cpu')
        if 'mae_visualize_vit_base' in args.weights_name:
            msg =  mae_model.load_state_dict(imagenet_checkpoint['model'], strict=True)
        elif 'mae_pretrain_vit_base' in args.weights_name:
            msg =  mae_model.load_state_dict(imagenet_checkpoint['model'], strict=False)
        else:
            raise ValueError(f'unknown {args.weights_name}')

        print(msg)  
    mae = mae_model
    

    assert isinstance(mae, torch.nn.Module), f"mae is {type(mae)}, not a model!"
    n_trainable = sum(p.requires_grad for p in mae.parameters())
    print(f"Trainable params: {n_trainable}")

    
    if args.lr is None:  
        args.lr = args.blr * args.eff_batch_size / 256

    optim_cfg = OptimCfg(lr=args.lr, weight_decay=args.weight_decay)

    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

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
        devices = args.num_devices,
        num_nodes=args.num_nodes,
        log_every_n_steps=25,
        strategy="ddp",   # 
        accumulate_grad_batches=args.accum_iter,   
        logger=wandb_logger,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        
        callbacks=[periodic_ckpt, last_ckpt]
    )

    trainer.fit(lit, train_loader, val_loader)
if __name__ == "__main__":
    main()
    
