from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from leverage_data.survival import (
    get_train_loader_surv,
    get_val_loader_surv,
    get_test_loader_surv,
    get_dataset,
    get_survival_head,
    Logger,
    get_encoder,
    train_survival_model,
    get_optimizer,
    Evaluate
)

import copy
import torch
import warnings
warnings.filterwarnings("ignore")


def verify_batchsize(train_size):
    batch_size = None
    if train_size is None:
        batch_size = 64
    elif train_size > 100 and train_size <= 5000:
        batch_size = 32
    elif train_size >= 10000:
        batch_size = 64
    elif train_size <= 100:
        batch_size = 8
    else:
        batch_size = np.nan
    return batch_size

def main():
    parser = ArgumentParser()
    parser.add_argument("--testrun", type=int, default=0)
    parser.add_argument("--freeze_encoder", type=int, default=None)
    # parser.add_argument("--dino_weights_pretrained", type=int, default=None)
    parser.add_argument("--enc_lr", type=float, default=1e-4)
    parser.add_argument("--mlp_wd", type=float, default=0.01)
    parser.add_argument("--mlp_lr", type=float, default=0.0004)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--estimator_to_use", type=str, default='breslow')
    parser.add_argument("--cloud", type=str, default='gber9')
    parser.add_argument("--prefix", type=str, default='R_')
    parser.add_argument("--dataset_name", type=str, default='areds_new')
    parser.add_argument("--finetuning_method", type=str, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--survival_head", type=str, default=None)
    parser.add_argument("--model_str", type=str, default='')
    parser.add_argument("--project_name", type=str, default='april_areds')
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--enc_opt", type=str, default="adamw",choices=["adamw", "adam", "sgd"], help="Optimizer type to use")
    parser.add_argument("--mlp_opt", type=str, default="adamw",choices=["adamw", "adam", "sgd"], help="Optimizer type to use")
    parser.add_argument("--metadata_csv", type=str, default="metadata.csv")
    # parser.add_argument("--use_encoder_weights", type=int, default=None, help="Path to encoder weights. None loads no pretrained weights.")
    parser.add_argument("--num_nodes", type=int, nargs='+', default=[64, 64], help="List of node counts for each layer in the MLP")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the MLP")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--crop_square_size", type=int, default=None, help="Size of the square circle crop to be applied to the images. If None, no circle crop is applied.")
    parser.add_argument("--train_size", type=int, default=32500, help="Number of training data to be used")
    parser.add_argument("--split_id", type=int, default=11042026)
    parser.add_argument("--use_more_transforms", type=int, default=None)
    parser.add_argument("--sweeprun", type=int, default=None)
    

    args = parser.parse_args()
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)

    hostname_ = args.cloud 
    dataset_name = args.dataset_name
    DATA_BASE_DIR = Path(__file__).resolve().parent
    config_file = DATA_BASE_DIR / "data/data_dirs.yaml"
    print('config file path', config_file)
    cfg = OmegaConf.load(config_file)
    project_dir = cfg['DATA'][hostname_][dataset_name]
    image_dir = f"{project_dir}"

    freeze_encoder = bool(args.freeze_encoder)
    in_features = 512

    args.testrun = bool(args.testrun)
    args.use_more_transforms = bool(args.use_more_transforms)
    

    args.batch_size = verify_batchsize(args.train_size)
    print(f'\n \n*********** {args.train_size}:{args.batch_size} ************* \n \n ')

    OPTIMIZERS = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

    encoder, in_features, weights_path_returned, backbone_model_str = get_encoder(args.weights_path, args.device, 
                                                                                   img_size=args.img_size, args = args)
    encoder = copy.deepcopy(encoder).to(args.device)

    if freeze_encoder is True:
        print("freezing encoder")
        for param in encoder.parameters():
            param.requires_grad = False
    else:
        for param in encoder.parameters():
            param.requires_grad = True
   
    
    mlp = get_survival_head(survival_head = args.survival_head, in_features=in_features,
                             num_nodes=args.num_nodes, dropout=args.dropout).to(args.device)
    # print(mlp)

    # Get data loaders
    train_loader  = get_train_loader_surv(
        image_dir=image_dir, metadata_csv=args.metadata_csv, 
        batch_size=args.batch_size, img_size=args.img_size, 
        crop_square_size=args.crop_square_size,
        train_size=args.train_size,
        split_id = args.split_id,
        disease_name = "diagnosis_amd_grade",
        prefix = args.prefix,
        more_augment = args.use_more_transforms,
    )
    val_loader = get_val_loader_surv(
        image_dir=image_dir, metadata_csv=args.metadata_csv, batch_size=args.batch_size, 
        img_size=args.img_size, crop_square_size=args.crop_square_size,
        split_id = args.split_id,
        disease_name = "diagnosis_amd_grade",
        prefix = args.prefix,
        more_augment = args.use_more_transforms,
    )
    test_loader = get_test_loader_surv(
        image_dir=image_dir, metadata_csv=args.metadata_csv, batch_size=args.batch_size, 
        img_size=args.img_size, crop_square_size=args.crop_square_size,
        split_id = args.split_id,
        disease_name = "diagnosis_amd_grade",
        prefix = args.prefix,
        more_augment = False,
    )

    # Get training data for Breslow estimation
    train_data = get_dataset(
        image_dir=image_dir, metadata_csv=args.metadata_csv, 
        split="train", img_size=args.img_size,
        n_train_data = args.train_size,
        split_id = args.split_id,
        disease_name = "diagnosis_amd_grade",
        prefix = args.prefix,
        augmentation = False,
    ).get_e_t()


    #including the mlp and encoder in the optimizer
    optimizer = get_optimizer([
                (mlp, OPTIMIZERS[args.mlp_opt], {"lr": args.mlp_lr, "weight_decay": args.mlp_wd}),
                (encoder, OPTIMIZERS[args.enc_opt], {"lr": args.enc_lr}),
            ])

    # Initialize logger
    config = vars(args)
    config['backbone_model_str'] = backbone_model_str
    config['weights_path_returned'] = weights_path_returned
    run_name = "testrun_" + str(bool(args.testrun)) + args.model_str + "_" + args.comment + "FE_" + str(freeze_encoder) + "_" + str(args.train_size) + "_"+ args.survival_head
    logger = Logger()
    tags = args.model_str
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.init(use_wandb=True, project_name=args.project_name,
                 config=config, tags=tags, run_name = run_name, 
                 timestamp= time_now)

    
    print('model string', args.model_str)
    # Train and evaluate model
    best_encoder, best_model, best_breslow = train_survival_model(
        survival_model=mlp,
        encoder=encoder,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        early_stopping_patience=args.early_stopping_patience,
        train_data=train_data,
        logger=logger,
        testrun=args.testrun,
        sweep=args.sweeprun,
        train_size = args.train_size,
        args = args)

    # Evaluate best model on val and test sets
    out, performance = Evaluate(
        survival_model=best_model,
        encoder=best_encoder,
        breslow=best_breslow,
        dataloaders={"val": val_loader, "test": test_loader},
        device=args.device,
        train_data=train_data,
        logger=logger,
        testrun=args.testrun,
        )
    f_path = f"reports_file.csv"
    print("performance in main",performance)
    logger.log_final_results(performance, file_path=f_path)

    
    print("\nThe model achieved the following performance on the test set:\n")
    print("Integrated Brier Score (IBS) (censoring adjusted):", performance["test"]["ibs"])
    print("Mean AUROC (censoring adjusted):", performance["test"]["mean_auroc"])
    print("AUROCs at the requested evaluation times (censoring adjusted):", performance["test"]["aurocs"])
    print("Concordance Index (C-index):", performance["test"]["concordance_index"])
    print("Done")




if __name__ == "__main__":
   
    main()
