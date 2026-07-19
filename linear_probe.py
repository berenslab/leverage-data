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
    parser.add_argument("--weights_path", type=str, default=None, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL', help='Name of model to train')
    args = parser.parse_args()
    print(args.device)

    start_time = datetime.datetime.now()
    start_time_fmt = start_time.strftime("%Y-%m-%d %H:%M:%S")

    config_file = '.secrets.yaml'
    cfg = OmegaConf.load(config_file)
    root = cfg['ROOT']
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

    
    all_data = load_nako(image_size =  args.img_size, 
                    batch_size = args.batch_size, 
                    label_name = None, 
                    augment_train = nako_transform_train, 
                    return_loader = False, 
                    augment_test = nako_transform_val,
                    augment_val = nako_transform_val,  
                    normalize = True, 
                    return_all = True, 
                    return_ids = True)
    all_dataset_train, _, _, _, transforms_ = all_data

    diseased_eyes_df =  pd.read_csv(os.path.join(root, 'datasets/NAKO/diseased_eyes.csv'))
    diseased_ids = list(diseased_eyes_df['ID'])

    change_lbls(all_dataset_train, diseased_ids)
    all_dataset_train.labels = all_dataset_train.labels.astype(int)

    print(np.unique(all_dataset_train.labels, return_counts = True))

    df = pd.DataFrame({'ID': all_dataset_train.ids, 'labels': all_dataset_train.labels})
    df['index'] = range(df.shape[0])

    print(df['labels'].value_counts())

    train, test_val = patient_id_split(df, "index", 
                                    split_by = 'ID', 
                                    test_split=0.4, seed=2021, 
                                    stratify_by = 'labels'
                                        ) 
    print(f"train {train.shape}, test_val {test_val.shape}")
    test, val = patient_id_split(test_val, "index", 
                                    split_by = 'ID', 
                                    test_split=0.5, seed=2021, 
                                    stratify_by = 'labels'
                                        ) 
    print(f"train: {train.shape}, test: {test.shape}, val: {val.shape}")

    train_list = list(train['index'].values)
    test_list = list(test['index'].values)
    val_list = list(val['index'].values)

    train_list = list(map(int, train_list))
    test_list = list(map(int, test_list))
    val_list = list(map(int, val_list))

    if args.testrun:
        train_list = random.sample(train_list, 20000)
        test_list = random.sample(test_list, 20000)
        val_list = random.sample(val_list, 20000)
        

    train_dataset = Subset(all_dataset_train, train_list)
    test_dataset  = Subset(all_dataset_train, test_list)
    val_dataset  = Subset(all_dataset_train, val_list)

    dataset_concat = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset ])#dataset_train, dataset_test, 
    train_loader_all = DataLoader(dataset_concat, batch_size = args.batch_size, shuffle = False)


    args.weights_path =  cfg['MODEL'][args.weights_path]
    finetune_weight = args.weights_path.split('/')[-3]
    #--------------------------------------------Model------------------------------------#

    model_light_chkpt = torch.load(args.weights_path, map_location='cpu', weights_only = False)
    mae = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    lit = MAELightning(mae)
    lit.load_state_dict(model_light_chkpt["state_dict"], strict=True)
    print('Loaded model')
    lit = lit.to('cuda')
    backbone = lit.mae

    for p in backbone.parameters():
        p.requires_grad = False

    features = predict_mae_transforms(backbone, train_loader_all, args.device)
    ids_, backbone_feat, label1 = features
    print(backbone_feat.shape, label1.shape, ids_.shape, np.unique(label1, return_counts=True))
    
    #-----------------------------------Linear Probing---------------------------------------------#
    
    acc, auc = linear_acc(X = backbone_feat, y = label1, id = ids_, id_split_name='patient_id', stratify_col='label' )
    print(f"acc {acc} auc {auc}")
    print("---- \n computing tsne \n ---")
    

    pca = PCA(n_components=100) 
    H_pca = pca.fit_transform(backbone_feat) 
    Y = TSNE(random_state=2000).fit(H_pca)

    exp_dir = f"results/FINETUNE/{args.dataset_name}/{args.task}/_{finetune_weight}_{transforms_}_{args.batch_size}_{start_time_fmt}"
    os.makedirs(exp_dir, exist_ok = True)
    np.savez(f"{exp_dir}/embeddings.npz",  Y = Y,  pcs = H_pca,  ids = ids_,
              H = backbone_feat, lbls1 = label1,
              )
    # assert backbone_feat.shape[1] == in_dim
    #----------------------------------------------------------------#
    
    plot_title = f'{args.dataset_name} '
    

    asyncio.run(plot_embeddings(ids_, Y, label1,
                    experiment_directory = exp_dir, 
                    plot_title = plot_title, 
                    stylef = cfg['STYLEF']
                            ))

    
if __name__ == "__main__":
    main()
 