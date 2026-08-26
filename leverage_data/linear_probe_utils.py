import os
import random
import pandas as pd
from PIL import Image
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


class NAKOBase(Dataset):
    def __init__(self, image_dir, metadata, transform=None):
        self.metadata = pd.read_csv(metadata)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        image_orient =  self.metadata['img_path'][idx]
        id = str(self.metadata['ID'][idx]) 
        image_path = os.path.join(self.image_dir, image_orient, id)
        image_path = f"{image_path}.jpg"

        image = Image.open(image_path)
        image = self.transform(image)
        label = self.metadata['any_eye_disease'][idx]
        return id, image, label, image_path


def extract_embeddings(model, dataloader, device = 'cuda',
                        testrun = False, encoder_only = False):
    """
    images: list or batch of images in HxWxC format (uint8 or float 0-1)
    returns: numpy array [num_images, num_patches, embed_dim]
    """
   
    model.to(device)
    model.eval()
    H_, labels_, ids_, paths_ = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            ids, imgs, labels, paths = batch
            imgs = imgs.to(device)
            embeddings = model(imgs)
            H_.append(embeddings.cpu().numpy())
            labels_.append(labels.numpy())
            ids_.append(ids)
            paths_.append(paths)


            if testrun and i == 2:  # for testing
                print(f"Batch {i} processed, breaking for testrun.")
                break

    H = np.concatenate(H_, axis = 0).astype(np.float16)
    labels11 = np.hstack(labels_).astype(np.uint8) 
    idss_ = np.hstack(ids_)
    all_paths = np.hstack(paths_)
    return H,  labels11, idss_, all_paths


def predict_mae_transforms(backbone, dataloader,
                            device="cuda", testrun=False):
    backbone.eval()
    features, labels, ids_, all_img_paths = [], [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            ids, x, y, image_path = batch
            x = x.to(device)
            feats = backbone.forward_encoder(x, mask_ratio=0)[0][:, 0]  # CLS token
            features.append(feats.cpu())
            labels.append(y)
            if isinstance(ids, (list, tuple)):
                if isinstance(ids[0], torch.Tensor):
                    ids_.append(torch.cat(ids))
                else:
                    ids_.extend(ids)
            else:
                ids_.append(ids)
            all_img_paths.append(image_path)
            if testrun and idx == 2:
                break
    if isinstance(ids_[0], torch.Tensor):
        id_out = torch.cat(ids_).numpy()
    else:
        id_out = np.array(ids_, dtype=object)
    all_img_paths = np.array(all_img_paths)
    return id_out, torch.cat(features).numpy(), torch.cat(labels).numpy(), all_img_paths


def change_lbls(dataset_, diseased_ids):
    for i, ids in enumerate(dataset_.ids):
        if ids in diseased_ids:
            dataset_.labels[i] = 1
        else:
            dataset_.labels[i] = 0