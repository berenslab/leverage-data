from .survival_loss import CoxPHLoss #neg_partial_log_likelihood #NLLDeepSurvLoss
from torchvision import models
import torch.nn as nn
import numpy as np
class MLP1Out(nn.Module):
    def __init__(self, in_features: int = 512, num_nodes=[32, 32], dropout: float = 0.1):
        super(MLP1Out, self).__init__()
        layers = []
        input_dim = in_features
        for output_dim in num_nodes:
            layers.append(DenseBlock(input_dim, output_dim, dropout))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

class LinearMLP(nn.Module):
    def __init__(self, in_features: int = 512):
        super(LinearMLP, self).__init__()
        # layers = []
        # input_dim = in_features
        # for output_dim in num_nodes:
        #     layers.append(DenseBlock(input_dim, output_dim, dropout))
        #     input_dim = output_dim
        # layers.append(nn.Linear(input_dim, 1, bias=False))
        self.model = nn.Linear(in_features, 1, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x


def create_encoder():
    """Create a feature extractor -- replace this function with loading your own model"""

    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.model = models.resnet18(weights=None)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.model.fc.in_features = in_features

        def forward(self, x):
            return self.model(x)

    return Encoder()

import torch
def get_survival_loss(log_partial_hazards, e, t, device = 'gpu'):
    """Get survival loss of batch. log_partial_hazards is a tensor of shape (batch_size, n_preds)
    e ("event indicator") and t ("time" / "duration to event") are tensors of shape (batch_size, )
    """
    # print(f"\n \n *********************** torch.unique(e).numel() {torch.unique(e)}********************* \n \n")
    assert torch.all((e == 0) | (e == 1)), "e must contain only 0 and 1. Did you swap event and time?"

    nll_loss = CoxPHLoss().to(device)
    nll_loss_ = nll_loss(log_partial_hazards, e, t)

    return nll_loss_


def get_survival_head(survival_head = None, in_features: int = 512, num_nodes=[64, 64], dropout: float = 0.1):
    if survival_head == 'mlp':
        return MLP1Out(in_features=in_features, num_nodes=num_nodes, dropout=dropout)
    elif survival_head == 'linear':
        return LinearMLP(in_features = in_features)