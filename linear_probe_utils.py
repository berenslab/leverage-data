
import torch
import numpy as np

def predict_mae_transforms(backbone, dataloader, device="cuda", testrun=False):
    backbone.eval()
    features, labels, ids_ = [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            ids, x, y = batch
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
            if testrun and idx == 2:
                break
    if isinstance(ids_[0], torch.Tensor):
        id_out = torch.cat(ids_).numpy()
    else:
        id_out = np.array(ids_, dtype=object)
    return id_out, torch.cat(features).numpy(), torch.cat(labels).numpy()


def change_lbls(dataset_, diseased_ids):
    for i, ids in enumerate(dataset_.ids):
        if ids in diseased_ids:
            dataset_.labels[i] = 1
        else:
            dataset_.labels[i] = 0