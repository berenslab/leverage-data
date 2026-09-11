import torch
import pickle
import random
import os, re
import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset
# import fundus_image_toolbox as fit
# Needed to prevent "OSError: image file is truncated"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(2025)
def rename_paths(path):
    return re.sub(r'\.(png|jpg|jpeg|PNG|JPG)$', '.jpg', path)

def split_by_event(df, x):

    # Step 1: Compute group proportions
    group_counts = df.groupby(['event', 'duration']).size()
    group_proportions = group_counts / group_counts.sum()

    # # Step 2: Determine sample size per group
    group_sample_counts = (group_proportions * x).round().astype(int)

    # # Step 3: Fix rounding mismatch
    diff = x - group_sample_counts.sum()
    if diff != 0:
        # Add/subtract the difference to the largest group
        adjust_idx = random.choice(group_sample_counts.index.tolist())
        group_sample_counts[adjust_idx] += diff
    # print('\n \n **** group_sample_counts *** \n', group_sample_counts)
    # # Step 4: Perform the sampling
    samples = []
    for (label1, label2), n in group_sample_counts.items():
        group_df = df[(df['event'] == label1) & (df['duration'] == label2)]
        # if len(group_df) < n:
        #     raise ValueError(f"Not enough samples for group {(label1, label2)}: needed {n}, available {len(group_df)}")
        samples.append(group_df.sample(n=n, random_state=42))

    sampled_df = pd.concat(samples).reset_index(drop=True)
    # print('sampled_df shape', sampled_df.shape)
    return sampled_df


class AredsSurvivalDataset(VisionDataset):
    """Pytorch Dataset for Areds fundus images"""

    def __init__(
        self,
        img_dir,
        metadata_csv,
        root="",
        transform=None,
        target_transform=None,
        crop_square_size=None,
        split="train",
        n_train_data = None,
        SPLIT_ID = "11-06-2025",
        label_name = "diagnosis_amd_grade_12c",
        patient_identifier = 'patient_id',
        prefix = 'C_',
        return_only_converters = False
    ):
        super(AredsSurvivalDataset, self).__init__(
            root, transform=transform, target_transform=target_transform # sets self.transform
        )

        self.split = split
        self.crop_square_size = crop_square_size
        self.img_dir = img_dir
        metadata_df = pd.read_csv(metadata_csv)
        self.n_train_data = n_train_data
        self.label_name = label_name
        self.return_only_converters = return_only_converters
        
        if self.return_only_converters:
            metadata_df = metadata_df[metadata_df[self.label_name] >= 10]
            print(f"Filtered dataset to only converters, new shape: {metadata_df.shape}")

        # Load patient_id sets for train, val, test splits at the given SPLIT_ID
        csv_dir = os.path.dirname(metadata_csv)
        split_ids_file = os.path.join(csv_dir, f"splits_patient-ids-{SPLIT_ID}.pkl")
        dict_seed_splits_patientids = pickle.load(open(split_ids_file, "rb"))
        print(f"Loaded split ids from {split_ids_file}")
        # print(f"Using split id {SPLIT_ID} for {split} split with {len(dict_seed_splits_patientids[SPLIT_ID][split])} patient ids.")
        
        if self.split == "train":
            train_all = metadata_df[metadata_df[patient_identifier].isin(dict_seed_splits_patientids[SPLIT_ID]["train"])]

            if self.n_train_data is None:
                train = train_all
            else:
                train = split_by_event(train_all, self.n_train_data)

            print('\n \n *** using n_train_data *** \n \n', train.shape)
            
            data = {"train": train}
        elif self.split == "val":
            val = metadata_df[metadata_df[patient_identifier].isin(dict_seed_splits_patientids[SPLIT_ID]["val"])]
            print('val shape', val.shape)
            data = {"val": val}
        elif self.split == "test":
            test = metadata_df[metadata_df[patient_identifier].isin(dict_seed_splits_patientids[SPLIT_ID]["test"])]
            print('test shape', test.shape)
            data = {"test": test}

        elif self.split == "train_subset":
            train_subset = metadata_df[metadata_df[patient_identifier].isin(dict_seed_splits_patientids[SPLIT_ID]["train_subset"])]
            if self.n_train_data is None:
                train_subset = train_subset
            else:
                train_subset = split_by_event(train_subset, self.n_train_data)

            print('train_subset shape', train_subset.shape)
            data = {"train_subset": train_subset}

        elif self.split == "val_subset":
            val_subset = metadata_df[metadata_df[patient_identifier].isin(dict_seed_splits_patientids[SPLIT_ID]["val_subset"])]
            print('val_subset shape', val_subset.shape)
            data = {"val_subset": val_subset}
        else:
            raise ValueError(f"split not recognised: {self.split}")
        
        print('split is', self.split)
        self._metadata_df = data[split].reset_index(drop=True)

        # Get the clf targets
        self._amd_grades = torch.LongTensor(
            self._metadata_df[self.label_name].values
        )

        # Get event indicators
        self._e = torch.LongTensor(self._metadata_df["event"].values)

        # Get time to event or censoring
        self._t = torch.LongTensor(self._metadata_df["duration"].values)

        # Get the image index in the dataset
        self._idx_array = torch.LongTensor(self._metadata_df.index.values)

    #     print('changing path names')
    #     self._input_array = [
    #     rename_paths(ele)  
    #     for ele in self._metadata_df["image_path"].values
    # ]
    #     self._input_array = [os.path.join(self.img_dir, f"{prefix}{ele}") for ele in self._input_array]
        # else:
        self._input_array = [
                os.path.join(self.img_dir, f"{prefix}{ele}") #ify
                for ele in self._metadata_df["image_path"].values
            ]
        
    def __len__(self):
        return len(self.amd_grades)

    def __getitem__(self, idx):
        """Returns an image, the amd grade, the event 
        indicator, the time to event or censoring and the image path."""
        
        grade = self.amd_grades[idx]

        e = self._e[idx].float()
        t = self._t[idx]

        path = self.get_img_path(idx)

        x = self.get_input(idx)

        return x, grade, e, t, path

    def get_e_t(self, idx: int = None):
        if idx is None:
            return self._e, self._t
        return self._e[idx], self._t[idx]

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
            - other (bool): If True, return the other image of a stereo pair
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        input_array = self._input_array

        img_filename = os.path.join(self.img_dir, input_array[idx])

        x = Image.open(img_filename).convert("RGB")

        if self.crop_square_size is not None:
            if not isinstance(self.crop_square_size, int):
                raise ValueError(
                    f"crop_square_size must be an int, got {type(self.crop_square_size)}"
                )
            try:
                x = Image.fromarray(fit.circle_crop.crop(x, size = (self.crop_square_size, self.crop_square_size)))
            except:
                print(f"[FIT] Error center cropping image {img_filename}. This img will not be centered.")
        
        if self.transform is not None:
            x = self.transform(x)

        return x

    def get_img_path(self, idx):
        return self._input_array[idx]

    @property
    def amd_grades(self):
        """
        A Tensor of grades, with amd_grades[i] representing the target of the i-th data point.
        """
        return self._amd_grades

    @property
    def event(self):
        """
        A Tensor of event indicators, with event[i] representing the event indicator of the i-th data point.
        """
        return self._e
    
    @property
    def time(self):
        """
        A Tensor of time to event or censoring, with time[i] representing the time to event or censoring of the i-th data point.
        """
        return self._t
