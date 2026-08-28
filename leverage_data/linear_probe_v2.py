import torch
import argparse
import asyncio
import datetime
import os, random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.plotting_utils import plot_embeddings
from sklearn.decomposition import PCA
from openTSNE import TSNE
from omegaconf import OmegaConf
from utils.helpers import linear_acc
from linear_probe_utils import NAKOBase, extract_embeddings
from models.load_models import get_encoder
from data.load_data import load_nako, load_new_nako
from multi_level_split.util import train_test_split as patient_id_split
from torch.utils.data import Subset

torch.set_float32_matmul_precision('medium')
random.seed(2024)

ENTITY = 'success_vera'
class LabeledSubset(Subset):
    @property
    def labels(self):
        return np.array(self.dataset.labels)[self.indices]
    
    @property
    def ids(self):
        return np.array(self.dataset.ids)[self.indices]

    @property
    def image_paths(self):
        return np.array(self.dataset.image_paths)[self.indices]


    
def change_lbls(dataset_, diseased_ids):
    for i, ids in enumerate(dataset_.ids):
        if ids in diseased_ids: 
            dataset_.labels[i] = 1
        else:
            dataset_.labels[i] = 0

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
    parser.add_argument("--testrun", type=int, default=0)
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
    dataset_root = cfg['DATASETS']['NAKO']['mlcloud']
    image_dir = os.path.join(dataset_root, 'images_lowres/224')

    weights_path =  cfg['MODEL']['mlcloud'][args.weights_name]

    nako_mean=[0.419, 0.209, 0.122]
    nako_std = [0.280, 0.164, 0.113]

    nako_transform_train = transforms.Compose([

        #     transforms.RandomResizedCrop(224,  scale=(0.2, 1.0)),
        #     transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = nako_mean, 
                                std = nako_std)
            ])


    # dataset_train_all = NAKOBase(image_dir,
    #                          f'{dataset_root}/NAKO_metadata/df_classify.csv',
    #                          transform=nako_transform_train)
    # train_loader_all = DataLoader(dataset_train_all, batch_size = args.batch_size, shuffle = False)
    # print(len(train_loader_all))
    all_data = load_nako( 
                        image_size = args.img_size, 
                        batch_size = args.batch_size, 
                        ssl_method = args.ssl_method,
                        return_loader=False,
                        normalize= True,
                        return_all = True,
                        augment_test=nako_transform_train,
                        augment_train=nako_transform_train
                                    )
    all_dataset_train, _, _, _, transforms_ = all_data

    print(type(all_dataset_train.labels))
    print(type(all_dataset_train.labels[0]))
    print(all_dataset_train.labels[0])

    diseased_eyes_df = pd.read_csv('/home/berens/bep973/ifeoma_home/data/NAKO/NAKO_metadata/old/diseased_eyes.csv')
    diseased_ids = list(diseased_eyes_df['ID'])
    change_lbls(all_dataset_train, diseased_ids)
    
    disease_labels = all_dataset_train.labels.astype(int)

    df = pd.DataFrame({'ID': all_dataset_train.ids, 'labels': disease_labels})
    df['index'] = range(df.shape[0])

    print(df['labels'].value_counts())
    print('labels',np.unique(disease_labels, return_counts = True))

    train, test_val = patient_id_split(df, "index", 
                                        split_by = 'ID', 
                                        test_split=0.2, seed=2021, 
                                        stratify_by = 'labels'
                                            ) 
    print(f"balanced datset pre-train shape {train.shape}, test shape {test_val.shape}")

    train_healthy = train[train['labels'] == 1].shape[0]
    test_healthy = test_val[test_val['labels'] == 1].shape[0]

    train1 = train[train['labels'] == 0].sample(n=train_healthy * 2)
    train2 = train[train['labels'] == 1]
    train_df = pd.concat([train1, train2])
    # print(train_df.shape)
    # train_df['labels'].value_counts()

    test1 = test_val[test_val['labels'] == 0].sample(n=test_healthy * 2)
    test2 = test_val[test_val['labels'] == 1]
    test_df = pd.concat([test1, test2])
    # print(test_df.shape)
    # test_df['labels'].value_counts()

    test_list = list(test_df['index'].values)
    test_list = list(map(int, test_list))

    train_list = list(train_df['index'].values)
    train_list = list(map(int, train_list))

    train_dataset = LabeledSubset(all_dataset_train, train_list)
    test_dataset  = LabeledSubset(all_dataset_train, test_list)

    print(f"train_dataset \n \n {np.unique(train_dataset.labels, return_counts=True)}")
    print(f"test_dataset \n \n {np.unique(test_dataset.labels, return_counts=True)}")


    dataset_concat = torch.utils.data.ConcatDataset([train_dataset, test_dataset ]) # dataset_train, dataset_test, 
    train_loader_all = DataLoader(dataset_concat, batch_size = args.batch_size, shuffle = False)
    

    encoder, in_features, weights_path_returned, backbone_model_str = get_encoder(weights_path, args.device, 
                                                                             img_size=args.img_size,)

    for p in encoder.parameters():
        p.requires_grad = False

    features = extract_embeddings(encoder, train_loader_all, args.device, testrun = args.testrun)

    backbone_feat, label1, ids_, paths_ = features
    print(backbone_feat.shape, label1.shape, ids_.shape, np.unique(label1, return_counts=True))

    acc, auc = linear_acc(X = backbone_feat, 
                          y = label1, 
                          id = ids_, 
                       stratify_col='label',
                        knn_classify=True )
    print(f"acc {acc} auc {auc}")
    print("---- \n computing tsne \n ---")


    pca = PCA(n_components=100) 
    H_pca = pca.fit_transform(backbone_feat) 
    Y = TSNE(random_state=2000).fit(H_pca)
    print('done tsne', Y.shape)
    exp_dir = f"results/{args.dataset_name}/{args.task}/{args.weights_name}/{args.batch_size}_{start_time_fmt}"
    os.makedirs(exp_dir, exist_ok = True)
   
   
    acc1, auc1 = linear_acc(X = Y, 
                          y = label1, 
                          id = ids_, 
                       stratify_col='label' )
    print(f"acc1 {acc1} auc1 {auc1}")
    pca_results = {'acc0': acc, 'auc0': auc, 'acc1': acc1, 'auc1': auc}


    plot_title = f'{args.dataset_name}'

    asyncio.run(plot_embeddings(pca_results, 
                                Y, 
                                label1,
                                experiment_directory = exp_dir, 
                    plot_title = plot_title, 
                    stylef = cfg['MAIN']['MLCLOUD']['STYLEF']
                            ))

    
if __name__ == "__main__":
    main()
 