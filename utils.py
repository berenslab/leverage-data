from scipy.spatial.distance import pdist
from types import SimpleNamespace
from torchvision import transforms
from PIL import ImageOps
import matplotlib as mpl
import numpy as np
import argparse
import random
import torch
from multi_level_split.util import train_test_split as patient_id_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


import os
from PIL import Image
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

    
def linear_acc(X, 
               y, 
                id, id_split_name = 'id', 
                target_label_name = 'label',
                SPLIT_SEED = 2026, 
                evaluate_weights = False, 
                stratify_col = None,
                knn_classify = False
                ):
    if stratify_col is None:
        stratify_col = target_label_name
        print(f"No stratification column provided using stratify col as {stratify_col}.")

    df = pd.DataFrame([id, y]).T
    df.columns = ['id', 'label']
  
    n_unique_labels = len(set(y))
    print('n unique labels',n_unique_labels )

    # df['label'] = y
    df['index'] = range(df.shape[0])
    train, test_val = patient_id_split(df, "index", 
                                       split_by = id_split_name, 
                                       test_split=0.2, seed=SPLIT_SEED, 
                                       stratify_by = stratify_col
                                         ) #'thickness_label'
    
    train_mask = train['index'].to_numpy()
    X_train = X[train_mask]
    y_train = y[train_mask]
    print(f"y train count {np.unique(y_train, return_counts = True)}")

    test_mask = test_val['index'].to_numpy()
    X_test = X[test_mask]
    y_test = y[test_mask]
    print(f"y test count {np.unique(y_test, return_counts = True)}")

    print(f"X train shape {X_train.shape} Y_train shape {y_train.shape} X_tests shape {X_test.shape} Y_test shape {y_test.shape}")

    if knn_classify: 
        acc_scores = []
        k_values = list(range(2, 20))
        for k in k_values:
            KNN = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(KNN, X_train, y_train,
                                    scoring='balanced_accuracy', cv=5)
            acc_scores.append(np.mean(scores))

        # Evaluate best k on untouched test set
        best_k = k_values[np.argmax(acc_scores)]
        KNN = KNeighborsClassifier(n_neighbors=best_k)
        KNN.fit(X_train, y_train)
        # use balanced accuracy for imbalanced datasets
        y_pred = KNN.predict(X_test)
        acc = round(balanced_accuracy_score(y_test, y_pred) * 100, 2)


        if n_unique_labels < 3:
            y_prob = KNN.predict_proba(X_test)[:, 1]  # probability of positive class
            auc = round(roc_auc_score(y_test, y_prob) * 100, 2)
        else:
            y_prob = KNN.predict_proba(X_test)
            auc = round(roc_auc_score(y_test, y_prob, multi_class='ovo') * 100, 2)

        return acc, auc, k_values, acc_scores
    
    else:
        if n_unique_labels < 3:
            lin = LogisticRegression( solver='lbfgs',  max_iter= 1000)

        else:
            lin = LogisticRegression(multi_class='multinomial', solver='lbfgs',  max_iter= 1000)

        lin.fit(X_train, y_train)
        # use balanced accuracy for imbalanced datasets
        y_pred = lin.predict(X_test)
        acc = round(balanced_accuracy_score(y_test, y_pred) * 100, 2)
        if n_unique_labels < 3:
            y_pred = lin.predict(X_test)
            auc = round(roc_auc_score(y_test, y_pred) * 100, 2)
        else:
            y_prob = lin.predict_proba(X_test)
            auc = round(roc_auc_score(y_test, y_prob, multi_class='ovo') * 100, 2)

        if evaluate_weights is True:
            return acc, auc, lin.coef_[0]
        else:
            return acc, auc
    


# def linear_acc(X, y, df, 
#                id_split_name, 
#                SPLIT_SEED = None, 
#                 evaluate_weights = False, stratify_col = None,):
#     n_unique_labels = len(set(y))
#     print('n unique labels',n_unique_labels )

#     # df['label'] = y
#     df['index'] = range(df.shape[0])
#     train, test_val = patient_id_split(df, "index", 
#                                        split_by = id_split_name, 
#                                        test_split=0.3, seed=SPLIT_SEED, 
#                                        stratify_by = stratify_col
#                                          ) #'thickness_label'
    
#     train_mask = train['index'].to_numpy()
#     X_train = X[train_mask]
#     y_train = y[train_mask]

#     test_mask = test_val['index'].to_numpy()
#     X_test = X[test_mask]
#     y_test = y[test_mask]

#     if n_unique_labels < 3:
#         lin = LogisticRegression( solver='lbfgs',  max_iter= 1000)

#     else:
#         lin = LogisticRegression(multi_class='multinomial', solver='lbfgs',  max_iter= 1000)

#     lin.fit(X_train, y_train)
#     acc = round(lin.score(X_test, y_test) * 100, 2)
#     if n_unique_labels < 3:
#         y_pred = lin.predict(X_test)
#         auc = round(roc_auc_score(y_test, y_pred) * 100, 2)
#     else:
#         y_prob = lin.predict_proba(X_test)
#         auc = round(roc_auc_score(y_test, y_prob, multi_class='ovo') * 100, 2)

#     if evaluate_weights is True:
#         return acc, auc, lin.coef_[0]
#     else:
#         return acc, auc
  

def prepare_model(models_mae, chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

augmentation_mapping = {
    'Resize': 'RS',
    'RandomApply': 'functrot', 
    'RandomRotation': 'randrot',
    'RandomResizedCrop': 'RC',
    'RandomHorizontalFlip': 'HF',
    'RandomVerticalFlip': 'VF',
    'ColorJitter': 'JI',
    'RandomGrayscale': 'GR',
    'Normalize': 'NR',
    'ToTensor': '' 
}

def aug2string(transforms_compose):
    short_strings = []
    for transform in transforms_compose.transforms:
        transform_name = type(transform).__name__
        if transform_name == 'RandomApply':
            for t in transform.transforms:
                if isinstance(t, transforms.ColorJitter):
                    short_strings.append(augmentation_mapping['ColorJitter'])
                else:
                    short_strings.append(augmentation_mapping.get(transform_name, 'unknown'))
        else:
            short_strings.append(augmentation_mapping.get(transform_name, 'unknown'))
    
    final_string = '_'.join(filter(None, short_strings))
    return final_string

def crop_from_center_coord(img, center_coord, crop_size):
    center_x, center_y = center_coord
    crop_width, crop_height = crop_size

    left = int(center_x - crop_width / 2)
    top = int(center_y - crop_height / 2)
    right = left + crop_width
    bottom = top + crop_height

    return img.crop((left, top, right, bottom))

def add_border(input_image, border_size, border_color):
        img_with_border = ImageOps.expand(input_image, border=border_size, fill=border_color)
        return img_with_border



labelcolors = np.array(
    [mpl.colors.to_hex("tab:orange"),
      mpl.colors.to_hex("tab:blue")]
)

def time_diff(time_):
    n = time_.shape[0]
    out_size = (n * (n - 1)) // 2
    dm = np.empty(out_size)
    k = 0
    for i in range(time_.shape[0]-1):
        for j in range(i+1, time_.shape[0]):
            diff = time_[j] - time_[i]
            dm[k] = diff
            k+=1
    return dm

def normalize(Z):
    z_norm = Z / np.sqrt((Z**2).sum(axis=1, keepdims=True)) 
    return z_norm

def corr_distance(repre, time, seq_length = 5):
    unique_points = int(repre.shape[0]/seq_length)

    dist_repre = []
    dist_time = []

    for i in range(unique_points):
        x_category = repre[i*seq_length:(i+1)*seq_length]
        y_category = time[i*seq_length:(i+1)*seq_length]
        
        dist_x = pdist(x_category, 'euclidean')
        dist_y = time_diff(y_category)
        
        dist_repre.append(dist_x)
        dist_time.append(dist_y)
        
    dist_repre_ = np.vstack(dist_repre)
    dist_time_ = np.vstack(dist_time)
    print(dist_repre_.shape, dist_time_.shape)

    return dist_repre_, dist_time_


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def args_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument("--n_encoder_heads", type=int, default=3)
    parser.add_argument("--pretrained", type=int, default=0)

    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument('--ex_dir', type = str, default = 'experiments', help = 'Path to save logs')
    parser.add_argument('--model_load_path', type=str, required=False, help = 'Path of saved model')
    parser.add_argument('--default_', type=str, default=True, help = 'whether to return the default set up of features or not')
    parser.add_argument('--plot_loss_curve', type=str, default=True, help = 'whether to plot_loss_curves or not')
    parser.add_argument('--n_epochs', type=int, nargs='*', default=[500, 50, 450 ], help='Number of training epochs')
    parser.add_argument('--n_epochs_', type=int, nargs='*', default=[1, 1, 1], help='Number of training epochs')
    parser.add_argument('--total_epochs', type=int,  default=100, help='Number of training epochs')
    parser.add_argument('--temperature', type=float,  default=None, help='temperature')
    parser.add_argument('--start_epoch', type=int, default=0, help = 'start epoch')
    parser.add_argument('--backbone', type=str, default='resnet18_', help = 'backbone model')
    parser.add_argument('--model_name', type=str, default=None, help = 'name of model: simclr, convnet, model')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
    parser.add_argument('--scheduler', type=str, default='CA_LR', help='scheduler')
    parser.add_argument('--ex_name', type=str, default='sar_five_longitudinal', help='experiment_name')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--sampling_scheme', type=str, default=None, help='sampling scheme for positive pairs')
    parser.add_argument('--seed', type=int, required=False, help='seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
    parser.add_argument('--seq_length', type=int, default=5, help='seq_length')
    parser.add_argument('--load_checkpoint', type=int, default=1, help='checkpoint path')
    parser.add_argument('--time_addition', type=int, default=1, help='time_addition')
    parser.add_argument('--longitude_index', type=int, default=0, help='whether to use the longitude index or the exact thickness value')
    parser.add_argument('--time_contrastive', type=int, default=0, help='whether to use a contrastive DS with time')
    parser.add_argument('--method', type=str, default='classifier',help='method')
    parser.add_argument('--dataset_name', type=str, default='MorphoMNIST', help='dataset_name')
    parser.add_argument('--pretrain', type=int, default=0, help='use a pretrained model')
    parser.add_argument('--comment', type=str, default='', help='comment about anything to note')
    parser.add_argument('--dataset_str', type=str, default='classic', help='classic, recons or other dataset name')
    parser.add_argument('--lambda_val', type=float, default=None, help='lambda_val')
    parser.add_argument('--scaling_val', type=float, default=None, help='scaling_val')
    parser.add_argument('--metric', type=str, default=None, help='type of metric')
    parser.add_argument('--ssl_method', type=str, default=None,help='ssl method: byol, simclr, tsimcne')
    parser.add_argument('--delta_type_', type=str, default= None, help='delta_type method: coswave, min_max')
    parser.add_argument('--label_type', type=str, default='thickness_label',help='delta_type method: coswave, min_max')
    parser.add_argument('--with_ids', type=str, default=False,help='determines whether or not you get ids. default true means no ids')
    parser.add_argument('--train_backbone', type=int, default=0, help='whether to use the longitude index or the exact thickness value')
    parser.add_argument('--use_loss_weights', type=int, default=0, help='whether to use the longitude index or the exact thickness value')
    parser.add_argument('--use_balance', default=None, help='Whether to use a balancing technique or not')
    parser.add_argument('--lossfn', default=None, help='name of loss function')
    parser.add_argument('--tmax', type=float, default=None, help='Whether to use a balancing technique or not')
    parser.add_argument('--batchnorm', type=bool, default=None, help='use batchnorm in MLP or not')
    parser.add_argument('--test', type=int, default=None, help='whether to use a small sample of the dataset or not')
    parser.add_argument('--update_margin_', type=int, default=None, help='use batchnorm in MLP or not')
    parser.add_argument('--margin_', type=float, default=None, help='use batchnorm in MLP or not')
    parser.add_argument('--s', type=float, default=None, help='use batchnorm in MLP or not')    

    # ---------- General ----------
    parser.add_argument("--name", type=str, default="vit")
    parser.add_argument("--ckpt_root", type=str, default="checkpoints")
    parser.add_argument("--pretrain_framework", type=str, choices=["simclr", "mae", "mlm", "clm"], default="simclr")
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--seed", type=int, default=2427633826)

    # ---------- Model settings ----------
    parser.add_argument("--use_time_embed", type=int, default=None, help='use use_time_embed or not')
    parser.add_argument("--learnable_time_embed", type=int, default=None, help='use learnable_time_embed or not')
    parser.add_argument("--relative_time_embed", action="store_true")
    parser.add_argument("--mlm_w", type=float, default=0.5)
    parser.add_argument("--mask_mode", type=str, default="v2")

    parser.add_argument("--use_ltn", type=int, default=None, help='use use_ltn or not')
    parser.add_argument("--temporal_dim", type=int, default=384)
    parser.add_argument("--temporal_depth", type=int, default=3)
    parser.add_argument("--interchangeable", action="store_true")
    parser.add_argument("--use_token_prediction_head", action="store_true")
    parser.add_argument("--use_variance_loss", action="store_true")

    # ---------- MLM params ----------
    parser.add_argument("--mlm_mask_ratio", type=float, default=0.15)

    # ---------- Solver ----------
    parser.add_argument("--vit_optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--vit_lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_epochs", type=int, default=20)

    # ---------- Trainer ----------
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--custom_save_ckpt_path", type=str, default="")
    parser.add_argument("--wandb_version", type=str, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=1)

    # ---------- Loader ----------
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")

    # ---------- SimCLR ----------
    parser.add_argument("--vit_simclr_temperature", type=float, default=0.5)

    # ---------- ViT model params ----------
    parser.add_argument("--model", type=str, default="vit")
    parser.add_argument("--vit_arch", type=str, choices=["tiny", "small", "base", "large"], default="small")
    parser.add_argument("--vit_embed_dim", type=int, default=384)
    parser.add_argument("--vit_patch_spatial", type=int, default=16)
    parser.add_argument("--vit_patch_temporal", type=int, default=1)
    parser.add_argument("--vit_drop_rate", type=float, default=0.1)
    parser.add_argument("--vit_drop_path_rate", type=float, default=0.0)
    parser.add_argument("--vit_image_only", action="store_true")

    # ---------- Data module ----------
    parser.add_argument("--data_target_class", type=str, default="datamodule.mnist_datamodule.MNISTDataModule")
    parser.add_argument("--data_data_name", type=str, default="variable")
    parser.add_argument("--data_name", type=str, default="mnist")  # dataset name

    parser.add_argument("--data_train_batch_size", type=int, default=64)
    parser.add_argument("--data_test_batch_size", type=int, default=64)
    parser.add_argument("--data_num_workers", type=int, default=4)
    parser.add_argument("--data_pin_memory", action="store_true")
    parser.add_argument("--data_persistent_workers", action="store_true")
    parser.add_argument("--data_img_size", type=int, default=128)
    parser.add_argument("--data_channels", type=int, default=1)
    parser.add_argument("--data_num_slices", type=int, default=1)
    parser.add_argument("--data_n_clips", type=int, default=2)
    parser.add_argument("--data_clip_frames", type=int, default=2)
    parser.add_argument("--data_stride", type=int, default=1)
    parser.add_argument("--data_data_percentage", type=float, default=1.0)
    parser.add_argument("--data_target", type=str, default=None)

    args = parser.parse_args()

    # ---------- Group args into namespaces ----------
    args.vit = SimpleNamespace(
        arch=args.vit_arch,
        embed_dim=args.vit_embed_dim,
        patch_spatial=args.vit_patch_spatial,
        patch_temporal=args.vit_patch_temporal,
        drop_rate=args.vit_drop_rate,
        drop_path_rate=args.vit_drop_path_rate,
        image_only=args.vit_image_only
    )

    args.data = SimpleNamespace(
        target_class=args.data_target_class,
        data_name=args.data_data_name,
        name=args.data_name,
        train_batch_size=args.data_train_batch_size,
        test_batch_size=args.data_test_batch_size,
        num_workers=args.data_num_workers,
        pin_memory=args.data_pin_memory,
        persistent_workers=args.data_persistent_workers,
        img_size=args.data_img_size,
        channels=args.data_channels,
        num_slices=args.data_num_slices,
        n_clips=args.data_n_clips,
        clip_frames=args.data_clip_frames,
        stride=args.data_stride,
        data_percentage=args.data_data_percentage,
        target=args.data_target
    )

    return args


def thickness(val):
    if val <= 0.8:
        return 0
    elif 0.8 < val <= 1.5:
        return 1
    elif 1.5 < val <= 2: 
        return 2
    elif 2 < val <=3:
        return 3
    elif 3 < val <= 4:
        return 4
    else:
        print(val)
        raise ValueError('unbounded value')
def thickness_old(val):
    if val <= 0.8:
        return 0
    elif 0.8 < val <= 1.5:
        return 1
    elif 1.5 < val <= 2: 
        return 2
    elif 2 < val <=3:
        return 3
    else:
        print(val)
        raise ValueError('unbounded value')

def get_backbone_output_dim(backbone, input_size=(3, 128, 128)):
    dummy_input = torch.randn(1, *input_size)
    print('dummy input shape', dummy_input.shape)
    with torch.no_grad():
        features = backbone(dummy_input)
    features = features.reshape(1, -1)
    print('getting backbone output shape', features.shape)
    return features.shape[1]
