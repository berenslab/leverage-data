# Dataloader
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms
import numpy as np

from survival_on_embedding.data.areds_survival_dataset import AredsSurvivalDataset
from survival_on_embedding.data.cnn_transforms import get_transforms
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_train_loader_surv(
                        image_dir: str, metadata_csv: str, 
                        batch_size: int,
                        img_size: int, 
                        num_workers: int=8, crop_square_size: int=None,
                        train_size: int = None,
                        split_id: str = '11-06-2025',
                        disease_name: str = "diagnosis_amd_grade_12c",
                        patient_identifier = 'patient_id',
                        prefix = 'C_',
                        more_augment = None,
                               ):
    """Get train dataloader

    Args:
        image_dir (str): Directory of images
        metadata_csv (str): Path to metadata CSV file
        batch_size (int): Batch size
        img_size (int): Image size
        augmentation (bool): Whether to apply augmentation
        num_workers (int): Number of workers

    Returns:
        train_loader (torch.utils.data.DataLoader)
    """

    train_set = get_dataset(split="train", image_dir=image_dir, 
                            metadata_csv=metadata_csv, 
                            img_size=img_size,
                            augmentation=more_augment, 
                            crop_square_size=crop_square_size,
                            n_train_data=train_size,
                            split_id = split_id,
                            disease_name = disease_name,
                            patient_identifier = patient_identifier,
                            prefix = prefix,
                            # transformations = transformations
                            )

    # # compute sample weights — give higher weight to event=1 rows
    # event_counts = df["event"].value_counts()
    # class_weights = 1.0 / event_counts
    # sample_weights = df["event"].map(class_weights).values

    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True
    # )

    # dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    # print(
    #     "The number of images in the training set is: ",
    #     len(train_loader) * batch_size,
    # )

    return train_loader

def get_train_subset_loader_surv(
                            image_dir: str, metadata_csv: str, 
                            batch_size: int,
                            img_size: int, 
                            num_workers: int=8, crop_square_size: int=None,
                            train_size: int = None,
                            split_id: str = '11-06-2025',
                            disease_name: str = "diagnosis_amd_grade_12c",
                            patient_identifier = 'patient_id',
                            prefix = 'C_',
                            more_augment = None,
                               ):
    """Get train dataloader

    Args:
        image_dir (str): Directory of images
        metadata_csv (str): Path to metadata CSV file
        batch_size (int): Batch size
        img_size (int): Image size
        augmentation (bool): Whether to apply augmentation
        num_workers (int): Number of workers

    Returns:
        train_loader (torch.utils.data.DataLoader)
    """

    train_subset = get_dataset(split="train_subset", image_dir=image_dir, 
                            metadata_csv=metadata_csv, 
                            img_size=img_size,
                            crop_square_size=crop_square_size,
                            n_train_data=train_size,
                            split_id = split_id,
                            disease_name = disease_name,
                            patient_identifier = patient_identifier,
                            prefix = prefix,
                            augmentation = more_augment,
                            )

    train_subset_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    
    )

    # print(
    #     "The number of images in the training set is: ",
    #     len(train_loader) * batch_size,
    # )

    return train_subset_loader


def get_val_loader_surv(image_dir: str, 
                        metadata_csv: str, batch_size: int,
                         img_size: int, num_workers: int=8, 
                         crop_square_size: int=None,
                         split_id: str = "11-06-2025",
                        disease_name: str = "diagnosis_amd_grade_12c",
                        patient_identifier = 'patient_id',
                        prefix = "C_",
                        more_augment = None,
                        ):
    """Get validation dataloader

    Args:
        image_dir (str): Directory of images
        metadata_csv (str): Path to metadata CSV file
        batch_size (int): Batch size
        img_size (int): Image size
        num_workers (int): Number of workers

    Returns:
        val_loader (torch.utils.data.DataLoader)
    """

    val_set = get_dataset(split="val", image_dir=image_dir, 
                            metadata_csv=metadata_csv, 
                            img_size=img_size, 
                            augmentation=more_augment, 
                            crop_square_size=crop_square_size,
                            split_id = split_id,
                            disease_name = disease_name,
                            patient_identifier = patient_identifier,
                            prefix = prefix,
                            # transformations = transformations
                        )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    
    )

    return val_loader

def get_val_subset_loader_surv(image_dir: str, metadata_csv: str, 
                                batch_size: int,
                                img_size: int, 
                                num_workers: int=8, crop_square_size: int=None,
                                train_size: int = None,
                                split_id: str = '11-06-2025',
                                disease_name: str = "diagnosis_amd_grade_12c",
                                patient_identifier = 'patient_id',
                                prefix = 'C_',
                                more_augment = None,
                               ):
    """Get val_subset dataloader

    Args:
        image_dir (str): Directory of images
        metadata_csv (str): Path to metadata CSV file
        batch_size (int): Batch size
        img_size (int): Image size
        augmentation (bool): Whether to apply augmentation
        num_workers (int): Number of workers

    Returns:
        val_subset_loader (torch.utils.data.DataLoader)
    """

    val_subset = get_dataset(split="val_subset", image_dir=image_dir, 
                            metadata_csv=metadata_csv, 
                            img_size=img_size,
                            augmentation=more_augment, 
                            crop_square_size=crop_square_size,
                            n_train_data=train_size,
                            split_id = split_id,
                            disease_name = disease_name,
                            patient_identifier = patient_identifier,
                            prefix = prefix,
                            # transformations=transformations
                            )

    val_subset_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

   

    return val_subset_loader

def get_test_loader_surv(image_dir: str, metadata_csv: str, 
                         batch_size: int, img_size: int, 
                         num_workers: int=8, crop_square_size: int=None,
                         split_id: str = "11-06-2025",
                        disease_name = "diagnosis_amd_grade_12c",
                        patient_identifier = 'patient_id',
                        prefix = 'C_',
                        more_augment = None
):
    """Get test dataloader

    Args:
        image_dir (str): Directory of images
        metadata_csv (str): Path to metadata CSV file
        batch_size (int): Batch size
        img_size (int): Image size
        num_workers (int): Number of workers

    Returns:
        test_loader (torch.utils.data.DataLoader)
    """

    test_set = get_dataset(split="test", image_dir=image_dir, 
                            metadata_csv=metadata_csv, 
                           img_size=img_size, 
                           augmentation=more_augment, 
                           crop_square_size=crop_square_size,
                            split_id = split_id,
                            disease_name = disease_name,
                            patient_identifier= patient_identifier,
                            prefix=prefix,
                            # transformations = transformations
                           )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    return test_loader

def get_dataset(split: str, image_dir: str,
                metadata_csv: str,
                img_size: int,
                augmentation: bool=None, 
                crop_square_size: int=None,
                n_train_data: int = None,
                split_id: str = None,
                disease_name: str = None,
                patient_identifier = 'patient_id',
                prefix = None,
                return_only_converters = False

                ):
    """Get dataset of a given split. 

    Args:
        split (str): train, val, or test
        image_dir (str): Directory of images
        metadata_csv (str): Path to metadata CSV file
        img_size (int): Image size
        augmentation (bool): Whether to apply augmentation
        pos_label (str): target label: all, late, wet or dry. If not passed, config entry is used.
    """
    # if transformations is None:
    transformations = get_transforms(img_size=img_size, more_augment=augmentation)

    # print(f'split is {split} and transformations is {transformations}')
    dataset = AredsSurvivalDataset(
        img_dir=image_dir,
        metadata_csv=metadata_csv,
        split=split,
        transform=transforms.Compose(transformations),
        crop_square_size=crop_square_size,
        n_train_data=n_train_data,
        SPLIT_ID=split_id,
        label_name=disease_name,
        patient_identifier = patient_identifier,
        prefix=prefix,
        return_only_converters=return_only_converters
    )

    return dataset

# def get_all_datasets(image_dir, metadata_csv, img_size, crop_square_size, train_size, batch_size):
#     print('train_size ---- \n \n ', train_size)
#     datasets = {
#         split: AredsSurvivalDataset(
#             img_dir=image_dir,
#             metadata_csv=metadata_csv,
#             # split=split,
#             transform=transforms.Compose(get_transforms(img_size=img_size, split="train" if split=="train" else "validation")),
#             crop_square_size=crop_square_size,
#             n_train_data=train_size if split == "train" else None
#         )
#         for split in ["train", "val", "test"]
#     }
#     train_loader = torch.utils.data.DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
#     val_loader = torch.utils.data.DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
#     test_loader = torch.utils.data.DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
#     return train_loader, val_loader, test_loader, datasets["train"]
