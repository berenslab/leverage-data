from json import encoder

import torch
import argparse
import asyncio
import datetime
import os, random
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from load_data import load_nako
from torch.utils.data import DataLoader
# from time_distance.MAE import models_mae
# from lightning_helpers import MAELightning
from torchvision import transforms
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from multi_level_split.util import train_test_split as patient_id_split
from linear_probe_utils import predict_mae_transforms, change_lbls
from plotting_utils import plot_embeddings, create_legend
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from openTSNE import TSNE
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from multi_level_split.util import train_test_split as patient_id_split
from utils import linear_acc
from linear_probe_utils import NAKOBase, extract_embeddings
from load_models import get_encoder

torch.set_float32_matmul_precision('medium')
random.seed(2024)

ENTITY = 'success_vera'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--ssl_method", type=str, default='')
    parser.add_argument("--dataset_name", type=str, default='nako')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--comment", type=str, default='')
    parser.add_argument("--group_name", type=str, default='TMI')
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--project", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--testrun", type=bool, default=False)
    parser.add_argument("--use_scheduler", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--weights_name", type=str, default=None, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL', help='Name of model to train')
    args = parser.parse_args()
    print(args.device)

    start_time = datetime.datetime.now()
    start_time_fmt = start_time.strftime("%Y-%m-%d %H:%M:%S")

    config_file = '.secrets.yaml'
    cfg = OmegaConf.load(config_file)
    dataset_root = cfg['DATASETS']['NAKO']['MLCLOUD']
    image_dir = os.path.join(dataset_root, 'images_lowres/224')

    weights_path =  cfg['MODEL']['mlcloud_original'][args.weights_name]

    nako_mean=[0.419, 0.209, 0.122]
    nako_std = [0.280, 0.164, 0.113]

    nako_transform_train = transforms.Compose([

            transforms.RandomResizedCrop(224,  scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = nako_mean, 
                                std = nako_std)
            ])
    nako_transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = nako_mean, 
                                std = nako_std)
            ])

    
    classify_metadata_df = pd.read_csv(f'{dataset_root}/NAKO_metadata/df_classify.csv')
    classify_metadata_df["index"] = range(classify_metadata_df.shape[0])

    dataset_train_all = NAKOBase(image_dir,
                             f'{dataset_root}/NAKO_metadata/df_classify.csv',
                             transform=nako_transform_train)
    train_loader_all = DataLoader(dataset_train_all, batch_size = args.batch_size, shuffle = False)

    encoder, in_features, weights_path_returned, backbone_model_str = get_encoder(weights_path, args.device, 
                                                                             img_size=args.img_size,)

    for p in encoder.parameters():
        p.requires_grad = False

    print(f"encoder {encoder} in_features {in_features} weights_path_returned {weights_path_returned} backbone_model_str {backbone_model_str}")
    features = extract_embeddings(encoder, train_loader_all, args.device, testrun = args.testrun)

    backbone_feat, label1, ids_, paths_ = features
    print(backbone_feat.shape, label1.shape, ids_.shape, np.unique(label1, return_counts=True))

    acc, auc = linear_acc(X = backbone_feat, 
                          y = label1, 
                          id = ids_, 
                       stratify_col='label' )
    print(f"acc {acc} auc {auc}")
    print("---- \n computing tsne \n ---")


    pca = PCA(n_components=100) 
    H_pca = pca.fit_transform(backbone_feat) 
    Y = TSNE(random_state=2000).fit(H_pca)
    print('done tsne', Y.shape)
    exp_dir = f"results/{args.dataset_name}/{args.task}/{args.batch_size}_{start_time_fmt}"
    os.makedirs(exp_dir, exist_ok = True)
    # np.savez(f"{exp_dir}/embeddings.npz",  Y = Y,  pcs = H_pca,  ids = ids_,
    #           H = backbone_feat, lbls1 = label1,
            #   )
    # assert backbone_feat.shape[1] == in_dim
    #----------------------------------------------------------------#
    
    plot_title = f'{args.dataset_name} '
    

    asyncio.run(plot_embeddings(ids_, 
                                Y, 
                                label1,
                                experiment_directory = exp_dir, 
                    plot_title = plot_title, 
                    stylef = cfg['STYLEF']
                            ))

    
if __name__ == "__main__":
    main()
 