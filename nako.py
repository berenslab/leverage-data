''' 
Functions and classes to load the NAKO dataset.
'''

import os
import glob
from pathlib import Path
import shutil
import json
import textwrap
import numpy as np
import random
from PIL import Image
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
# import fundus_image_toolbox as fit

config_file = '../../.secrets.yaml'
cfg = OmegaConf.load(config_file)
# print(cfg['DATASETS']['CIN'])
root = cfg['DATASETS']['CIN']['nako_main']

# Variables of the NAKO dataset
NAKO_DIR = {'590': f'{root}'}
IMAGE_TYPES = ['rt_leftcentral', 'rt_leftnasal', 'rt_rightcentral', 'rt_rightnasal']
EYE_DISEASES = ['d_an_aug_1', 'd_an_aug_2', 'd_an_aug_3', 'd_an_metdm2c_2', 
                'd_an_metdm2c_1','d_an_met_1']
SEED = 42
NORMALIZATION = {'mean': [0.419, 0.209, 0.122], 'std': [0.270, 0.157, 0.105]}


def get_nako_paths(dataset='590', baseline_followup='baseline', image_dir='images_lowres', img_res='224'):
    '''
    Get the paths for the files of the NAKO dataset. Modify this function for different folder structures.
    Args:
        dataset (str): version of the NAKO dataset, use "590" or "810".
        baseline_followup (str): "baseline" of "followup1".
        image_dir (str): "images_lowres" or "images" if the original images are to be loaded. 
        img_res (str): image resolution to choose the appropriate image directory, 
            not necessary when loading the original images.
    '''

    nako_dir = NAKO_DIR[dataset]
    if dataset == '810':
        nako_dir = os.path.join(nako_dir, baseline_followup)
        
    project_dir = Path.cwd() # Temporary: quality results are stored in the repo
    quality_dir = project_dir.joinpath('quality_results', f'{dataset}_{baseline_followup}')
    quality_dir.mkdir(parents=True, exist_ok=True)

    nako_paths = {'nako_dir': nako_dir,
                  'image_dir': os.path.join(nako_dir, image_dir),
                  'lowres_dir': os.path.join(nako_dir, 'images_lowres'),
                  'metadata_file': os.path.join(nako_dir, 'NAKO_metadata','labels', 'metadata_translated.csv'),
                  'labels_file': os.path.join(nako_dir, 'NAKO_metadata','labels', 'labels.csv'),
                  'quality_dir': quality_dir}
    
    if image_dir == 'images_lowres':
        nako_paths['image_dir'] = os.path.join(nako_paths['image_dir'], str(img_res))

    # Temporary: metadata for 810 is not yet available
    nako_paths['metadata_file'] = f'{root}/NAKO_metadata/labels/metadata_translated.csv'

    return nako_paths


class FeatureDecoder:
    '''
    Retrieves metadata for the features of the NAKO dataset.

    Includes feature descriptions and mappings from categorical values to 
    human-readable labels.
    '''
    def __init__(self, metadata_file):
        self.df_meta = pd.read_csv(metadata_file)
        self.df_meta[self.df_meta.columns[3]] = self.df_meta[self.df_meta.columns[3]].map(lambda x: str(x).replace("'", ""))

    def decode(self, feature_name):
        '''Get mapping and human-readable label from a specific feature.'''
        labels = self.df_meta[self.df_meta['Variablenname'] == feature_name]
        # print('labels in feature deacider', labels.value_counts())
        # print(labels.head())
        label = labels['Label'].iloc[0]
        labels = labels[labels.columns[2:]]
        # print('another labels', labels)
        mapping = dict(zip(labels[labels.columns[0]], labels[labels.columns[1]]))
        # print('mapping', mapping)
        return mapping, label
    
    def group_nans(self, feature_name, mapping, df=None):
        '''Group missing values under a single nan value if they exist for a feature.'''
        missing = []
        for k, v in mapping.items():
            if ('missing' in v.lower()) or ('unbekannt' in v.lower()) or ('nan' in v.lower()):
                if df is not None:
                    df[feature_name] = df[feature_name].replace(k, np.nan)
                missing.append(k)   
        if missing:
            mapping[np.nan] = 'Missing'
            _ = [mapping.pop(k) for k in missing]
        return df, mapping

    def modify_categorical(self, feature_name, mapping, df=None):
        '''Replace arbitrary categorical codes with integers from 0 to n_categories.'''
        is_nan = mapping.pop(np.nan, None)
        new_mapping = {np.nan: 'Missing'} if is_nan else {}
        
        for i, (k, v) in enumerate(mapping.items()):
            
            new_mapping[i] = v
            if df is not None:
                df[feature_name] = df[feature_name].replace(k, i)

        return df, new_mapping
    
    def shorten_mapping(self, mapping, width=15):
        '''Truncate labels in a mapping to have a specific width.'''
        for k,v in mapping.items():
            mapping[k] = textwrap.shorten(v, width=width, placeholder='...')
        return mapping
    
    def wrap_mapping(self, mapping, width=15):
        '''Wrap labels in a mapping to have a specific width by adding new lines.'''
        for k,v in mapping.items():
            mapping[k] = '\n'.join(textwrap.wrap(v, width=width))
        return mapping


class NakoDataset(Dataset):
    '''
    Loads images and labels from NAKO pre-processed dataset.
    '''
    def __init__(self, nako_paths, image_type, transform=None, 
                 split='train', 
                 kfold=0, feature_name=None, drop_nan=False,
                   return_ids = False, contrastive = False,
                   filter_kwargs={}):
        '''
        Args:
            nako_paths (dict): paths for the nako repository, obtained by using get_nako_paths().
            image_types (str, list[str]): image type(s) to load.
            transform (object): image transform.
            split (str): "train", "val", "test" or "all" for kfold=0, otherwise use "train" or "test" only.
            kfold (int): integer from 0 to 5. Use 1 to 5 to get the 5-fold splits and 0 to get train-val-test split.
            feature_name (str or list[str]): names of features to retrieve from the labels file.
            drop_nan (bool): whether to drop nan values when retrieving a feature.
            filter_kwargs (dict): feature_names, value and any_all ("any" or "all") to filter dataset.
        '''
        self.nako_paths = nako_paths
        self.image_type = image_type if isinstance(image_type, list) else [image_type]
        self.transform = transform
        self.decoder = FeatureDecoder(metadata_file=self.nako_paths['metadata_file'])
        self.return_ids = return_ids
        self.contrastive = contrastive

        # Get participant ids for the chosen split
        self.ids = self.get_ids(split, kfold)

        # Get specific feature
        if feature_name:
            self.feature_name = feature_name
            self.ids, self.labels, self.feature_label, self.mapping = self.get_feature(self.ids, feature_name, drop_nan, filter_kwargs)
        else:
            self.labels = self.ids # Labels are in this case the patient ids

        # Get paths and corresponding features for each id
        self.ids, self.labels, self.image_paths = self.get_image_paths()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # print('image path in nako', self.image_paths[idx])
        image = Image.open(self.image_paths[idx])
        if not self.contrastive:
            if self.transform:
                image = self.transform(image)
        label = self.labels[idx]

        if self.return_ids and not self.contrastive:
            ids_ = self.ids[idx]
            return ids_, image, label
        
        if self.return_ids and self.contrastive:
            image1  = self.transform(image)
            image2 =  self.transform(image)
            ids_ = self.ids[idx]
            return ids_, image1, image2, label, self.image_paths[idx]


        return image, label

    def get_ids(self, split, kfold):
        '''Get participant ids for the chosen split.'''
        assert (((kfold == 0) and (split in ["train", "val", "test", "all"])) or ((kfold in list(range(1, 6))) and (split in ["train", "val", "test"]))), 'Invalid combination of split and kfold.'
        
        splits_file = os.path.join(self.nako_paths['lowres_dir'], 'splits.json')
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        if split == 'all':
            ids = np.array(splits['0']['train'] + splits['0']['val'] + splits['0']['test'])
        elif split == 'test':
            ids = np.array(splits['0']['test'])
        else:
            ids = np.array(splits[str(kfold)][split])
        
        ids.sort()
        return ids

    def get_feature(self, ids, feature_names, drop_nan=False, filter_kwargs={}):
        '''Get feature values for specific participant ids from the labels file.'''
        feature_names  = feature_names if isinstance(feature_names, list) else [feature_names]

        # Get labels
        df = pd.read_csv(self.nako_paths['labels_file'], index_col='ID', usecols=['ID'] + feature_names + filter_kwargs.get('feature_names', []))
        df = df.loc[ids] # Filter to only get requested records

        df, mappings, feature_labels = self.get_metadata(feature_names, df, drop_nan)
        df = df.dropna(subset=feature_names) if drop_nan else df
        df = self.filter_dataset(df, **filter_kwargs) if filter_kwargs else df
        
        features = df[feature_names].to_numpy(dtype=np.float32)
        features = features.squeeze() if len(feature_names) == 1 else features
        ids = df.index.to_numpy()

        return ids, features, feature_labels, mappings
    
    def get_metadata(self, feature_names, df, drop_nan):
        '''Fetch metadata for each feature and pre-process labels (group nans and process categorical features).'''
        mappings, feature_labels = [], []
        for feature_name in feature_names:
            mapping, feature_label = self.decoder.decode(feature_name)
            # print(f"mapping {mapping} feature_label {feature_label}")
            df, mapping = self.decoder.group_nans(feature_name, mapping, df) # If it contains missing fields group them

            # df, mapping = self.decoder.modify_categorical(feature_name, mapping, df) #ify removed this
            mapping = self.decoder.wrap_mapping(mapping)

            if drop_nan:
                mapping.pop(np.nan, None)

            mappings.append(mapping)
            feature_labels.append(feature_label)

        feature_labels = feature_labels[0] if len(feature_names) == 1 else feature_labels
        mappings = mappings[0] if len(feature_names) == 1 else mappings
        return df, mappings, feature_labels
    
    def filter_dataset(self, df, feature_names, value=0, any_all='any'):
        '''Filter dataset based on a specific feature.'''
        mask = (df[feature_names] == value)
        mask = mask.any(axis=1) if any_all == 'any' else mask.all(axis=1)
        df = df[mask]
        return df
    
    def get_image_paths(self):
        mapping_file = os.path.join(self.nako_paths['lowres_dir'], 'mapping.csv')
        df = pd.read_csv(mapping_file, index_col='ID')
        df = df.loc[self.ids]

        all_ids, all_labels, all_paths = [], [], []
        for image_type in self.image_type:
            mask = df[image_type].to_numpy()
            ids = self.ids[mask].tolist()
            labels = self.labels[mask].tolist()
            paths = [os.path.join(self.nako_paths['image_dir'], image_type, f'{i}.jpg') for i in ids]

            all_ids.extend(ids)
            all_labels.extend(labels)
            all_paths.extend(paths)

        return np.array(all_ids), np.array(all_labels, dtype=np.float32), np.array(all_paths)


class NakoContrastiveDataset(NakoDataset):
    '''
    Loads images from the same eye as positive pairs and labels from the NAKO pre-processed dataset.
    '''
    def __init__(self, nako_paths, image_type, transform=None, split='all', feature_name=None, drop_nan=False, filter_kwargs={}):
        self.nako_paths = nako_paths
        self.image_type = image_type
        self.transform = transform
        self.decoder = FeatureDecoder(metadata_file=self.nako_paths['metadata_file'])

        # Get participant ids
        self.ids = self.get_ids(split, 0)

        # Filter to ids with existing images of both types
        self.ids = self.filter_ids(self.ids)

        # Get specific feature
        if feature_name:
            self.feature_name = feature_name
            self.ids, self.labels, self.feature_label, self.mapping = self.get_feature(self.ids, feature_name, drop_nan, filter_kwargs)
        else:
            self.labels = self.ids # Labels are in this case the patient ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = []
        for image_type in self.image_type:
            image_path = os.path.join(self.nako_paths['image_dir'], image_type, f'{self.ids[idx]}.jpg')
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        label = self.labels[idx]
        
        return (tuple(images), label)

    def filter_ids(self, ids):
        '''Get participant ids with existing images of every type.'''
        mapping_file = os.path.join(self.nako_paths['lowres_dir'], 'mapping.csv')
        df = pd.read_csv(mapping_file, index_col='ID')
        df = df.loc[ids]
        df = df[df[self.image_type].all(axis=1)]
        ids = df.index.to_numpy()
        return ids


class NakoImageDataset(NakoDataset):
    '''
    Loads images from a subdirectory of the raw NAKO dataset.
    '''
    def __init__(self, image_dir, nako_paths, unusable_images=None, transform=None, feature_name=None, drop_nan=False):
        '''
        Args:
            nako_paths (dict): paths for the nako repository, obtained by using get_nako_paths().
                Used to fetch labels and metadata.
            image_dir (str): subdirectory of the raw NAKO dataset.
            unusable_images (list[str]): list of images to be excluded.
            transform (object): image transform.
        '''
        self.image_dir = image_dir
        self.nako_paths = nako_paths
        self.transform = transform
        self.decoder = FeatureDecoder(metadata_file=nako_paths['metadata_file'])

        # Get participant ids
        self.ids = self.get_ids(image_dir, unusable_images)
        
        # Get specific feature
        if feature_name:
            self.feature_name = feature_name
            self.ids, self.labels, self.feature_label, self.mapping = self.get_feature(self.ids, feature_name, drop_nan)
        else:
            self.labels = self.ids # Labels are in this case the patient ids

        self.image_paths = [os.path.join(self.image_dir, f'{i}.jpg') for i in self.ids]

    def get_ids(self, image_dir, unusable_images):
        '''Get participant ids with existing images in image_dir.'''
        image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
        image_paths = ['/'.join(p.split(sep='/')[-2:]) for p in image_paths]

        # Discard unusable images
        if unusable_images:
            image_paths = list(set(image_paths).difference(set(unusable_images)))

        # Get patient ids
        ids = [int(os.path.basename(p).split(sep='.')[0]) for p in image_paths]
        ids = np.array(ids)
        ids.sort()
        return ids


def load_healthy_diseased(nako_paths, image_type, transform, feature_name):
    '''Load NAKO dataset and filter it to get only healthy or diseased individuals.'''
    decoder = FeatureDecoder(nako_paths['metadata_file'])
    mapping, _ = decoder.decode(EYE_DISEASES[0])
    _, mapping = decoder.group_nans(EYE_DISEASES[0], mapping)
    # _, mapping = decoder.modify_categorical(EYE_DISEASES[0], mapping) #ify, removed this
    mapping_r = dict(zip(mapping[-1].values(), mapping[-1].keys()))
    
    # Get healthy train and test splits and diseased split
    filter_kwargs = {'feature_names': EYE_DISEASES, 'value': mapping_r['No'], 'any_all': 'all'}
    dataset_train = NakoDataset(nako_paths['image_dir'], image_type, transform=transform, split='train', feature_name=feature_name, drop_nan=True, filter_kwargs=filter_kwargs)
    dataset_val = NakoDataset(nako_paths['image_dir'], image_type, transform=transform, split='val', feature_name=feature_name, drop_nan=True, filter_kwargs=filter_kwargs)
    dataset_test = NakoDataset(nako_paths['image_dir'], image_type, transform=transform, split='test', feature_name=feature_name, drop_nan=True, filter_kwargs=filter_kwargs)
    filter_kwargs = {'feature_names': EYE_DISEASES, 'value': mapping_r['Yes'], 'any_all': 'any'}
    dataset_diseased = NakoDataset(nako_paths['image_dir'], image_type, transform=transform, split='all', feature_name=feature_name, drop_nan=True, filter_kwargs=filter_kwargs)

    return dataset_train, dataset_val, dataset_test, dataset_diseased

        
def get_unusable_images(quality_dir):
    '''Get a list of unusable images (corrupted, lowres, ungradable) from the NAKO dataset.'''
    # List of corrupted images
    corrupted_file = quality_dir.joinpath('corrupted_images.json')
    corrupted_images = []
    if corrupted_file.exists():
        with open(corrupted_file, 'rb') as f:
            corrupted_images = json.load(f)

    # List of lowres images
    lowres_file = quality_dir.joinpath('lowres_images.json')
    lowres_images= []
    if lowres_file.exists():
        with open(lowres_file, 'rb') as f:
            lowres_images = json.load(f)

    # List of ungradable images
    ungradable_file = quality_dir.joinpath('ungradable_images.json')
    ungradable_images = []
    if ungradable_file.exists():
        with open(ungradable_file, 'rb') as f:
            ungradable_images = json.load(f)

    return corrupted_images, lowres_images, ungradable_images


def fundus_circle_crop(nako_paths, image_size=(512, 512)):
    '''Crop fundus images from the NAKO dataset using the fundus image toolbox.'''
    corrupted_images, low_res_images, _ = get_unusable_images(nako_paths['quality_dir'])
    transform = transforms.ToPILImage()

    for image_type in IMAGE_TYPES:
        for i in range(3):
            print(image_type, i + 1)
            image_dir = os.path.join(nako_paths['image_dir'], image_type + '_' + str(i+1))
            dataset_dir = Path(os.path.join(nako_paths['lowres_dir'], str(image_size[0]) + '_raw', f'{image_type}_{str(i+1)}'))
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset = NakoImageDataset(image_dir, nako_paths, corrupted_images + low_res_images)

            for image, idx in dataset:
                outputs = fit.crop(image, size=image_size)
                image = transform(outputs)
                image.save(dataset_dir.joinpath(dataset_dir, f'{idx}.jpg'))


def clean_dataset(quality_dir, lowres_image_dir, image_size=512):
    '''Create a version of the NAKO dataset without ungradable images and only with the best quality image per field per participant. '''
    quality_dir = quality_dir.joinpath('fundus_image_toolbox')
    source_dir = os.path.join(lowres_image_dir, str(image_size) + '_raw')
    target_dir = os.path.join(lowres_image_dir, str(image_size))
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    for image_type in IMAGE_TYPES:
        print(image_type)
        target_subdir = Path(target_dir).joinpath(image_type)
        target_subdir.mkdir(parents=True, exist_ok=True)
        df_list = []
        for i in range(1, 4):
            image_subdir = image_type + '_' + str(i)
            quality_file = quality_dir.joinpath(image_subdir + '.csv')
            df = pd.read_csv(quality_file, dtype={'ID': str}, index_col='ID')
            df = df[df['label'] == 1]   # Filter to only gradable images
            df['dir'] = image_subdir
            df.rename(columns={'confidence': image_subdir}, inplace=True)
            df_list.append(df[[image_subdir]])

        # Choose the folder with the best quality image per ID
        df_combined = pd.concat(df_list, axis=1)
        df_combined['best_dir'] = df_combined.idxmax(axis=1)

        print(f'Copying {df_combined.shape[0]} images to {target_subdir}...')
        source_files = (source_dir + '/' + df_combined['best_dir'] + '/' + df_combined.index + '.jpg').to_list()
        [shutil.copy(f, target_subdir) for f in source_files]


def resize_dataset(nako_paths, source_dir, target_dir, image_size):
    '''Create a resized version of the NAKO dataset.'''
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.Resize(image_size)])
    dataset = NakoImageDataset(source_dir, nako_paths, transform=transform)

    for image, idx in tqdm(dataset):
        image.save(os.path.join(target_dir, f'{idx}.jpg'))


def generate_image_mapping(nako_paths):
    '''Create a csv file with participant ids that indicates if they have images for each eye/field.'''
    mapping_file = os.path.join(nako_paths['lowres_dir'], 'mapping.csv')

    df = pd.read_csv(nako_paths['labels_file'], index_col='ID', usecols=['ID'])
    all_ids = []
    for image_type in IMAGE_TYPES:
        image_dir = os.path.join(nako_paths['lowres_dir'], '512', image_type)
        image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
        ids = np.array([int(os.path.basename(p).split(sep='.')[0]) for p in image_paths])
        all_ids.extend(ids)
        df[image_type] = False
        df.loc[ids, image_type] = True

    df = df.loc[np.array(list(set(all_ids)))]
    df.sort_index(inplace=True)
    df.to_csv(mapping_file, mode='w', index=True, header=True)


def generate_splits(nako_paths):
    '''Create a json file with train-val-test and 5-fold splits for the participant ids.'''
    #  Get a list of ids with existing gradable images
    mapping_file = os.path.join(nako_paths['lowres_dir'], 'mapping.csv')
    df = pd.read_csv(mapping_file, index_col='ID')
    ids = df.index.to_numpy()

    splits = {}
    # Generate train, val and test splits (80-10-10)
    ids_train, ids_test = train_test_split(ids, test_size=0.2, shuffle=True, random_state=SEED)
    ids_val, ids_test = train_test_split(ids_test, test_size=0.5, shuffle=True, random_state=SEED)
    splits[0] = {'train': ids_train.tolist(), 'val': ids_val.tolist(), 'test': ids_test.tolist()}

    # Generate 5 fold splits
    ids = np.concat((ids_train, ids_val), axis=0)
    kfold = KFold(5, shuffle=True, random_state=SEED)
    for i, (train_idx, val_idx) in enumerate(kfold.split(ids), start=1):
        splits[i] = {'train': ids[train_idx].tolist(), 'val': ids[val_idx].tolist()}

    splits_file = os.path.join(nako_paths['lowres_dir'], 'splits.json')
    with open(splits_file, 'w') as f:
        json.dump(splits, f)


def generate_splits_combined(nako_paths_baseline, nako_paths_followup):
    '''Create a json file with train-val-test and 5-fold splits for the participant ids of the baseline and followup.'''
    #  Get a list of ids with existing gradable images
    mapping_file = os.path.join(nako_paths_baseline['lowres_dir'], 'mapping.csv')
    df = pd.read_csv(mapping_file, index_col='ID')
    ids_baseline = df.index.to_numpy()

    mapping_file = os.path.join(nako_paths_followup['lowres_dir'], 'mapping.csv')
    df = pd.read_csv(mapping_file, index_col='ID')
    ids_followup = df.index.to_numpy()
    all_ids = np.concatenate((ids_baseline, ids_followup), axis=0)

    splits_baseline, splits_followup = {}, {}
    for ids, splits in zip([ids_baseline, ids_followup], [splits_baseline, splits_followup]):
        # Generate train, val and test splits (80-10-10)
        ids_train, ids_test = train_test_split(all_ids, test_size=0.2, shuffle=True, random_state=SEED)
        ids_val, ids_test = train_test_split(ids_test, test_size=0.5, shuffle=True, random_state=SEED)
        splits[0] = {'train': np.intersect1d(ids_train, ids).tolist(), 
                    'val': np.intersect1d(ids_val, ids).tolist(), 
                    'test': np.intersect1d(ids_test, ids).tolist()}

        # Generate 5 fold splits
        ids_train = np.concat((ids_train, ids_val), axis=0)
        kfold = KFold(5, shuffle=True, random_state=SEED)
        for i, (train_idx, val_idx) in enumerate(kfold.split(ids_train), start=1):
            splits[i] = {'train': np.intersect1d(ids_train[train_idx], ids).tolist(), 
                         'val': np.intersect1d(ids_train[val_idx], ids).tolist()}

    # Save splits file for both dataset partitions
    splits_file = os.path.join(nako_paths_baseline['lowres_dir'], 'splits.json')
    with open(splits_file, 'w') as f:
        json.dump(splits_baseline, f)

    splits_file = os.path.join(nako_paths_followup['lowres_dir'], 'splits.json')
    with open(splits_file, 'w') as f:
        json.dump(splits_followup, f)


def get_dataset_normalization(nako_paths):
    '''Compute normalization parameters for the NAKO dataset.'''
    n_channels = 3
    dataset = NakoDataset(nako_paths, IMAGE_TYPES, transforms.ToTensor(), 'train')
    
    all_mean, all_std = np.zeros((len(dataset), n_channels)), np.zeros((len(dataset), n_channels))
    for i, (image, _) in enumerate(dataset):
        all_mean[i, :] = image.mean(axis=(1,2)).numpy()
        all_std[i, :] = image.std(axis=(1,2)).numpy()

    return all_mean.mean(axis=0), all_std.mean(axis=0)
    

def get_site_ids(nako_paths, site='Augsburg'):
    '''Get participant ids from a specific site.'''
    decoder = FeatureDecoder(nako_paths['metadata_file'])
    mapping, _ = decoder.decode('basis_uort')
    mapping_r = dict(zip(mapping.values(), mapping.keys()))

    # Get lables from metadata file
    df = pd.read_csv(nako_paths['label_file'], usecols=['ID', 'basis_uort'])
    df = df[df['basis_uort'] == mapping_r[site]]

    return df['ID'].to_numpy()


# class DrawCircle(object):
#     '''Draw circle at a specific coordinate of an image.'''
#     import cv2

#     def __init__(self, center_coord=(140, 180), radius=5, color=(255, 255, 255), thickness=-1):
#         self.center_coord = center_coord
#         self.radius = radius
#         self.color = color
#         self.thickness = thickness
        
#     def __call__(self, image):
#         image = np.array(image)
#         img = cv2.circle(image, self.center_coord, self.radius, self.color, self.thickness)
#         return img
    
#     def __repr__(self):
#         return self.__class__.__name__+'()'
    

# def distort_dataset(image_dir, distorted_ids):
#     '''Add distortion to a subset of images of the dataset, to simulate a lens artifact.'''
#     transform = transforms.Compose([DrawCircle(), transforms.ToPILImage()])
#     dataset = NakoImageDataset(image_dir)

#     distorted_idx = get_matching_index(dataset.ids, distorted_ids)

#     for idx in distorted_idx:
#         image = dataset[idx][0]
#         image = transform(image)
#         image.save(image_dir.joinpath(image_dir, f'{dataset.ids[idx]}.jpg'))


def get_matching_index(arr_ref, arr_match):
    '''Get indices for elements in arr_ref that match elements in arr_match.'''
    df = pd.DataFrame({'Value': np.arange(arr_ref.size)}, index=arr_ref)
    arr_match = arr_match[np.isin(arr_match, arr_ref)]
    df = df.loc[arr_match]
    return df['Value'].to_numpy()


def flip_dataset(source_dir, target_dir):
    '''Create a flipped version of the NAKO dataset.'''
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
    dataset = NakoImageDataset(source_dir, transform=transform)

    for image, idx in tqdm(dataset):
        image.save(os.path.join(target_dir, f'{idx}.jpg'))


def change_csv_encoding(source_file, dest_file):
    '''Change encoding of csv file from the german default to english.'''
    # Encoding for german characters: https://stackoverflow.com/questions/62872697/how-to-open-a-german-csv-file-with-pandas
    image_types_ab = ['rt_leftcent', 'rt_leftnasal', 'rt_rightcent', 'rt_rightnasal']
    if '810' in source_file:
        df_labels = pd.read_csv(source_file, index_col='ID', sep=';', encoding='latin1')
    else:
        date_conv = lambda x: pd.to_datetime(x, format='%Y-%m-%d')
        date_cols = ['{}_{}_date'.format(image_type, j) for image_type in image_types_ab for j in range(1,4)]
        df_labels = pd.read_csv(source_file, index_col='ID', sep=';', converters=dict.fromkeys(date_cols, date_conv), encoding='latin1')

    df_labels.to_csv(dest_file, mode='w', index=True, header=True)


def add_row_csv(source_file, dest_file, index_col='ID', row_idx=0, fill=np.nan):
    df = pd.read_csv(source_file, index_col=index_col)
    df.loc[row_idx] = fill
    df = df.sort_index()
    df.to_csv(dest_file, mode='w', index=True, header=True)


def load_cifar10(transform):
    '''Load cifar10 dataset, to run tests.'''
    dataset_train = CIFAR10(root="datasets", train=True, download=True, transform=transform)
    dataset_test = CIFAR10(root="datasets", train=False, transform=transform)
    mapping = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
    return dataset_train, dataset_test, mapping


def sort_categorical_df(df, column_name, ascending=False):
    '''Sort dataframe based on the counts for each category of a column that holds a categorical variable.'''
    df['sort_col'] = df[column_name].map(df[column_name].value_counts())
    df = df.sort_values(by='sort_col', ascending=ascending).drop('sort_col', axis=1)
    return df


def load_image(filename, return_np=True):
    '''Load image file as a PIL image or a numpy array.'''
    image = Image.open(filename)
    image.load()
    data = np.asarray(image, dtype='int32')
    return data if return_np else image


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


# if __name__ == '__main__':
#     print('nako.py')