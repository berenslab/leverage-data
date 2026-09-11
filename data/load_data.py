
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from .nako import NakoDataset, get_nako_paths
from sklearn.model_selection import train_test_split
from multi_level_split.util import train_test_split as patient_id_split
import pandas as pd  
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional
import torch


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
class LoadData:
    pass


class AREDSVanillaDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        metadata_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        return_all = 'img',
        prefix = 'C_',
        zero_index = None,
        **kwargs
    ) -> None:

        self.image_dir = image_dir
        self.metadata_df = pd.read_csv(metadata_df)
        self.transform = transform
        self.return_all = return_all
        self.prefix = prefix
        self.zero_index = zero_index
        self.options = kwargs
        assert self.zero_index != None, "please, check the labels of your data and write the correct zero_index (must be 0 or 1)"

        
    def __getitem__(self, idx: int):
        row= self.metadata_df.iloc[idx]
        
        patient_id = row["eye_id"]
        label1 =  row["diagnosis_amd_grade"]
        if self.zero_index == 0:
            print('[WARNING]: using indexed labels where 0 corresponds to the lowest grade of AMD, and 11 corresponds to the highest grade of AMD')
            label = label1
        elif self.zero_index == 1:
            print('[WARNING]: using indexed labels where 1 corresponds to the lowest grade of AMD, and 12 corresponds to the highest grade of AMD')
            label = label1 - 1
        # label2 =  row["converters"]
        image_path = str(row["image_path"])
        patient_age = row["patient_age"]
        patient_sex = row["patient_sex"]
        
        img_filename = os.path.join(self.image_dir, f"{self.prefix}{image_path}")
        # print(patient_id, label, image_path)
        image = Image.open(img_filename).convert("RGB")

        if self.transform is not None:
            image1 = self.transform(image)
            image2 = self.transform(image)
        else:
            return patient_id, image, label


        if self.return_all == 'img':
            return patient_id, image1, image2, label
        elif self.return_all == 'all':
            return patient_id, image1, image_path, img_filename,  label, patient_age, patient_sex
        elif self.return_all == "img_lbl":
            return image1, label
        # elif self.return_all == 'converters':
        #     return image1, label2
        elif self.return_all == 'preti':
            return torch.stack([image1, image2], dim=0)
        elif self.return_all == 'metadata':
            return patient_id, image1, image2, label1, patient_age, patient_sex
            # return patient_id, image1, image2, label1, label2, patient_age, patient_sex
        else:
            return image1, image2


    def __len__(self):
        return self.metadata_df.shape[0]

def load_nako( image_size = 224, batch_size = 512, 
              augment_train = None, 
              label_name = 'basis_uort', 
              contrastive = False, return_loader = True, 
              augment_test = None, augment_val = None,
              normalize = True, return_all = False, 
              return_ids = False,
              IMAGE_TYPES = ['rt_leftcentral', 'rt_leftnasal', 'rt_rightcentral', 'rt_rightnasal'],
              **kwargs):
    image_dir = get_nako_paths(dataset='590', img_res=image_size)



    # base_image_dir = os.path.join(LOWRES_IMAGES_DIR, str(224), 'rt_leftcentral') # Only for nako

    if augment_train is None:
        if normalize:
            augment_train = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(image_size), 
                                        #  transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),  
                                        transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomGrayscale(p=0.2),
                                        transforms.Normalize(mean=[0.419, 0.209, 0.122],
                                                             std = [0.280, 0.164, 0.113] )
                                        # transforms.Normalize(mean = [0.417, 0.201, 0.114],
                                        #                      std =  [0.265, 0.140, 0.090])
                                         
                                         ])
        else:
            augment_train = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(image_size), 
                                        #  transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),  
                                        transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomGrayscale(p=0.2),
                                         ])
    transforms_ = f"{aug2string(augment_train)}"  

    if augment_test is None:
        augment_test = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    if augment_val is None:
        augment_val = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])



    dataset_train = NakoDataset(image_dir, IMAGE_TYPES, transform=augment_train, split='train', return_ids=return_ids, feature_name=label_name, contrastive=contrastive)
    dataset_test = NakoDataset(image_dir, IMAGE_TYPES, transform=augment_test, split='test', return_ids=return_ids, feature_name=label_name, contrastive=contrastive)
    dataset_val = NakoDataset(image_dir, IMAGE_TYPES, transform=augment_val, split='val', return_ids=return_ids, feature_name=label_name, contrastive=contrastive)

    train_loader = DataLoader(dataset_train, batch_size = batch_size, num_workers = 8, shuffle= True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=8, shuffle=False)  
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=8, shuffle=False) 
    
    print(f"Dataset  loaded with {len(dataset_train)} training samples, {len(dataset_test)} test samples, and {len(dataset_val)} validation samples.")

    if return_all:
        print('returning all')
        dataset_train_all = NakoDataset(image_dir, IMAGE_TYPES, transform=augment_train, split='all', return_ids=return_ids, feature_name=label_name, contrastive=contrastive)
        train_loader_all = DataLoader(dataset_train_all, batch_size = batch_size, num_workers = 20, shuffle= True)
        if return_loader is not True:
            print('returning all datasets')
            return dataset_train_all, dataset_train, dataset_test, dataset_val, transforms_
        else: 
            print('returning all dataloaders')
            return train_loader_all, train_loader, test_loader, (val_loader, dataset_val), transforms_

    
    if return_loader is not True:
        return dataset_train, dataset_test, dataset_val, transforms_
    print(f"return loader  is {return_loader} therefore returning ify")
    return train_loader, test_loader, (val_loader, dataset_val), transforms_


def load_new_nako( image_size = 224, augment_train = None, normalize = True, ):
    
    train_meta = load_datasets('nako_new')
    train_ssl, vanilla_train, val, test,  image_folder = train_meta

    if augment_train is None:
        if normalize:
            augment_train = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(image_size), 
                                        #  transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),  
                                        transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomGrayscale(p=0.2),
                                        transforms.Normalize(mean=[0.419, 0.209, 0.122],
                                                             std = [0.280, 0.164, 0.113] )
                                        # transforms.Normalize(mean = [0.417, 0.201, 0.114],
                                        #                      std =  [0.265, 0.140, 0.090])
                                         
                                         ])
        else:
            augment_train = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(image_size), 
                                        #  transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),  
                                        transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomGrayscale(p=0.2),
                                         ])
    
    dataset_train = NAKONewDataset(image_folder, train_ssl, transform = augment_train)
    return dataset_train
