from typing import Union

import numpy as np
import pandas as pd
import torch
from sksurv.util import check_y_survival
from sksurv.nonparametric import CensoringDistributionEstimator
from survival_on_embedding.common.conversion_utils import et_tuple_to_df, e_t_to_tuple
def check_ipcw_calc(y_train_set, y_val_set):
    """ Check that IPCW is calculated. Will throw error if not. 
    
    Args:
        y_train_set: train set ordered array or tuple of (event, time)
        y_val_set: val set, s.o."""

    y_train = (y_train_set[0].to("cpu").numpy(), y_train_set[1].to("cpu").numpy())
    y_train = e_t_to_tuple(y_train[0], y_train[1])

    y_val = (y_val_set[0].to("cpu").numpy(), y_val_set[1].to("cpu").numpy())
    y_val = e_t_to_tuple(y_val[0], y_val[1])

    cde = CensoringDistributionEstimator()
    event, time = check_y_survival(y_train)
    event_val, time_val = check_y_survival(y_val)
    cde.fit(y_train)
    print("max time train", max(time_val))
    kaplan_probas = cde.predict_proba(time_val[event_val])
    cde.predict_ipcw(y_val)

    return True



def get_event_indicator_matrix(events: Union[torch.Tensor, np.ndarray], durations: Union[torch.Tensor, np.ndarray]):
    """Matrix with 1s at all visits >= duration to event. Times are columns, patients are rows.
    Column index corresponds to time in the unit given by the model's durations/times array.
    
    Example: times = [0, 1, 2, 3, 4, 5]
                labels are then found through: calc_event_indicator_matrix()[:, times] 
    """

    m = torch if isinstance(events, torch.Tensor) else np

    visits = m.arange(0, durations.max()+1)
    y_ = m.zeros((len(events), len(visits)))

    for i in range(len(events)):
        for v in visits:
            if v >= durations[i]:
                if events[i] == 1.0:
                    y_[i, v] = 1.0
    
    return y_

def get_cuda(gpu: Union[str, int] = ""):
    """Returns cuda string for given GPU

        e.g. 'cuda:1' or 'cuda'

    Args:
        gpu (str, otional): gpu number(s)
    Returns:
        str
    """
    gpu = str(gpu)
    if gpu == "":
        return "cuda"
    else:
        return f"cuda:{gpu}"


def set_parameter_requires_grad(self, fine_tune: bool = True):
    if fine_tune:
        for param in self.parameters():
            param.requires_grad = True
    elif not fine_tune:
        for param in self.parameters():
            param.requires_grad = False


def optimizer_to_device(optim, device: str) -> None:
    """Move optimizer to gpu device

    Args:
        optim: Optimizer
        device (str): Device to move to

    Returns:
        None
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def get_lr(optimizer):
    for p in optimizer.param_groups:
        return p["lr"]


def get_optimizer(modules_and_opts):
   
    optimizers = []
    for module, opt_class, opt_kwargs in modules_and_opts:
        optimizer = opt_class(module.parameters(), **opt_kwargs)
        optimizers.append(optimizer)
    return optimizers
