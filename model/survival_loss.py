import numpy as np
import pandas as pd
import sys
import os

import torch
import torch.nn as nn

from .breslow_estimator import BreslowEstimator


# def log_substract(x, y):
#     """log(exp(x) - exp(y))"""
#     return x + torch.log1p(-(y - x).exp())

# if hasattr(torch.Tensor, "scatter_reduce_"):
#     # version >= 1.12
#     def scatter_reduce(input, dim, index, reduce, *, output_size=None):
#         src = input
#         if output_size is None:
#             output_size = index.max() + 1
#         return torch.empty(output_size, device=input.device).scatter_reduce(
#             dim=dim, index=index, src=src, reduce=reduce, include_self=False
#         )

# else:
#     scatter_reduce = torch.scatter_reduce

# def scatter_logsumexp(input, index, *, dim=-1, output_size=None):
#     """Inspired by torch_scatter.logsumexp
#     Uses torch.scatter_reduce for performance
#     """
#     max_value_per_index = scatter_reduce(
#         input, dim=dim, index=index, output_size=output_size, reduce="amax"
#     )
#     max_per_src_element = max_value_per_index.gather(dim, index)
#     recentered_scores = input - max_per_src_element
#     sum_per_index = scatter_reduce(
#         recentered_scores.exp(),
#         dim=dim,
#         index=index,
#         output_size=output_size,
#         reduce="sum",
#     )
#     return max_value_per_index + sum_per_index.log()

# class CoxPHLoss(torch.nn.Module):
#     """Loss for CoxPH model."""

#     allowed = ("breslow", "efron")

#     def __init__(self, method = "efron"):
#         super().__init__()
#         assert method in self.allowed, f"Method must be one of {self.allowed}"
#         self.method = method

#     def forward(self, log_h, events, durations):
#         log_h = log_h.flatten()

#         # durations, events = y.T

#         # sort input
#         durations, idx = durations.sort(descending=True)
#         log_h = log_h[idx]
#         events = events[idx]

#         event_ind = events.nonzero().flatten()
#         if event_ind.nelement() == 0:
#             # return 0 while connecting the gradient
#             return log_h[:0].sum()

#         # numerator
#         log_num = log_h[event_ind].mean()

#         # logcumsumexp of events
#         event_lcse = torch.logcumsumexp(log_h, dim=0)[event_ind]

#         # number of events for each unique risk set
#         _, tie_inverses, tie_count = torch.unique_consecutive(
#             durations[event_ind], return_counts=True, return_inverse=True
#         )

#         # position of last event (lowest duration) of each unique risk set
#         tie_pos = tie_count.cumsum(axis=0) - 1

#         # logcumsumexp by tie for each event
#         event_tie_lcse = event_lcse[tie_pos][tie_inverses]

#         if self.method == "breslow":
#             log_den = event_tie_lcse.mean()

#         elif self.method == "efron":
#             # based on https://bydmitry.github.io/efron-tensorflow.html

#             # logsumexp of ties, duplicated within tie set
#             tie_lse = scatter_logsumexp(log_h[event_ind], tie_inverses, dim=0)[
#                 tie_inverses
#             ]
#             # multiply (add in log space) with corrective factor
#             aux = torch.ones_like(tie_inverses)
#             aux[tie_pos[:-1] + 1] -= tie_count[:-1]
#             event_id_in_tie = torch.cumsum(aux, dim=0) - 1
#             discounted_tie_lse = (
#                 tie_lse
#                 + torch.log(event_id_in_tie)
#                 - torch.log(tie_count[tie_inverses])
#             )

#             # denominator
#             log_den = log_substract(event_tie_lcse, discounted_tie_lse).mean()

#         # loss is negative log likelihood
#         return log_den - log_num


# import sys
# import warnings

# import torch

# from torchsurv.tools.validate_data import validate_model, validate_survival_data
# Adapted from https://github.com/Novartis/torchsurv/blob/main/src/torchsurv/loss/cox.py


# __all__ = [
#     "_partial_likelihood_cox",
#     "_partial_likelihood_efron",
#     "_partial_likelihood_breslow",
#     "neg_partial_log_likelihood",
# ]


# def _partial_likelihood_cox(
#     log_hz_sorted: torch.Tensor,
#     event_sorted: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Args:
#         log_hz_sorted (torch.Tensor, float): Log hazard rates sorted by time.
#         event_sorted (torch.Tensor, bool): Binary tensor indicating if the event occurred (True) or was censored (False), sorted by time.

#     Returns:
#         torch.Tensor: partial log likelihood for the Cox proportional hazards model in the absence of ties in event time.
#     """
#     event_sorted = event_sorted.bool()
#     log_hz_flipped = log_hz_sorted.flip(0)
#     log_denominator = torch.logcumsumexp(log_hz_flipped, dim=0).flip(0)
#     # print('\n \n ************ event_sorted **************** \n \n', event_sorted)
#     # print('\n \n ************ event_sorted dtype**************** \n \n', event_sorted.dtype)
#     return (log_hz_sorted - log_denominator)[event_sorted]


# def _partial_likelihood_efron(
#     log_hz_sorted: torch.Tensor,
#     event_sorted: torch.Tensor,
#     time_sorted: torch.Tensor,
#     time_unique: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Args:
#         log_hz_sorted (torch.Tensor, float): Log hazard rates sorted by time.
#         event_sorted (torch.Tensor, bool): Binary tensor indicating if the event occurred (True) or was censored (False), sorted by time.
#         time_sorted (torch.Tensor, float): Event or censoring times sorted in ascending order.
#         time_unique (torch.Tensor, float): Event or censoring times sorted without ties.

#     Returns:
#         torch.Tensor: partial log likelihood for the Cox proportional hazards model using Efron's method to handle ties in event time.
#     """
#     J = len(time_unique)

#     # H = [torch.where((time_sorted == time_unique[j]) & (event_sorted))[0] for j in range(J)]
#     H = [torch.where((time_sorted == time_unique[j]) & (event_sorted.bool()))[0]for j in range(J)]
#     R = [torch.where(time_sorted >= time_unique[j])[0] for j in range(J)]

#     # Calculate the length of each element in H and store it in a tensor
#     m = torch.tensor([len(h) for h in H])

#     # Create a boolean tensor indicating whether each element in H has a length greater than 0
#     include = torch.tensor([len(h) > 0 for h in H])

#     log_nominator = torch.stack([torch.sum(log_hz_sorted[h]) for h in H])

#     denominator_naive = torch.stack([torch.sum(torch.exp(log_hz_sorted[r])) for r in R])
#     denominator_ties = torch.stack([torch.sum(torch.exp(log_hz_sorted[h])) for h in H])

#     log_denominator_efron = torch.zeros(J, device=log_hz_sorted.device)
#     for j in range(J):
#         mj = int(m[j].item())
#         for sample in range(1, mj + 1):
#             log_denominator_efron[j] += torch.log(
#                 denominator_naive[j] - (sample - 1) / float(m[j]) * denominator_ties[j]
#             )
#     return (log_nominator - log_denominator_efron)[include]


# def _partial_likelihood_breslow(
#     log_hz_sorted: torch.Tensor,
#     event_sorted: torch.Tensor,
#     time_sorted: torch.Tensor,
# ):
#     """
#     Compute the partial likelihood using Breslow's method for Cox proportional hazards model.

#     Args:
#         log_hz_sorted (torch.Tensor, float): Log hazard rates sorted by time.
#         event_sorted (torch.Tensor, bool): Binary tensor indicating if the event occurred (True) or was censored (False), sorted by time.
#         time_sorted (torch.Tensor, float): Event or censoring times sorted in ascending order.

#     Returns:
#         torch.Tensor: partial likelihood for the observed events.
#     """  # noqa: E501
#     event_sorted = event_sorted.bool()
#     N = len(time_sorted)
#     R = [torch.where(time_sorted >= time_sorted[i])[0] for i in range(N)]
#     log_denominator = torch.stack([torch.logsumexp(log_hz_sorted[R[i]], dim=0) for i in range(N)])

#     return (log_hz_sorted - log_denominator)[event_sorted]


# def neg_partial_log_likelihood(
#     log_hz: torch.Tensor,
#     event: torch.Tensor,
#     time: torch.Tensor,
#     ties_method: str = "efron",
#     reduction: str = "mean",
#     checks: bool = True,
# ) -> torch.Tensor:
#     r"""Compute the negative of the partial log likelihood for the Cox proportional hazards model.

#     Args:
#         log_hz (torch.Tensor, float):
#             Log relative hazard of length n_samples.
#         event (torch.Tensor, bool):
#             Event indicator of length n_samples (= True if event occurred).
#         time (torch.Tensor):
#             Event or censoring time of length n_samples.
#         ties_method (str):
#             Method to handle ties in event time. Defaults to "efron".
#             Must be one of the following: "efron", "breslow".
#         reduction (str):
#             Method to reduce losses. Defaults to "mean".
#             Must be one of the following: "sum", "mean".
#         checks (bool):
#             Whether to perform input format checks.
#             Enabling checks can help catch potential issues in the input data.
#             Defaults to True.

#     Returns:
#         (torch.tensor, float):
#             Negative of the partial log likelihood.

#     Note:
#         For each subject :math:`i \in \{1, \cdots, N\}`, denote :math:`X_i` as the survival time and :math:`D_i` as the
#         censoring time. Survival data consist of the event indicator, :math:`\delta_i=1(X_i\leq D_i)`
#         (argument ``event``) and the time-to-event or censoring, :math:`T_i = \min(\{ X_i,D_i \})`
#         (argument ``time``).

#         The log hazard function for the Cox proportional hazards model has the form:

#         .. math::

#             \log \lambda_i (t) = \log \lambda_{0}(t) + \log \theta_i

#         where :math:`\log \theta_i` is the log relative hazard (argument ``log_hz``).

#         **No ties in event time.**
#         If the set :math:`\{T_i: \delta_i = 1\}_{i = 1, \cdots, N}` represent unique event times (i.e., no ties),
#         the standard Cox partial likelihood can be used :cite:p:`Cox1972`. Let :math:`\tau_1 < \tau_2 < \cdots < \tau_N`
#         be the ordered times and let  :math:`R(\tau_i) = \{ j: \tau_j \geq \tau_i\}`
#         be the risk set at :math:`\tau_i`. The partial log likelihood is defined as:

#         .. math::

#             pll = \sum_{i: \: \delta_i = 1} \left(\log \theta_i - \log\left(\sum_{j \in R(\tau_i)} \theta_j \right) \right)

#         **Ties in event time handled with Breslow's method.**
#         Breslow's method :cite:p:`Breslow1975` describes the approach in which the procedure described above is used unmodified,
#         even when ties are present. If two subjects A and B have the same event time, subject A will be at risk for the
#         event that happened to B, and B will be at risk for the event that happened to A.
#         Let :math:`\xi_1 < \xi_2 < \cdots` denote the unique ordered times (i.e., unique :math:`\tau_i`). Let :math:`H_k` be the set of
#         subjects that have an event at time :math:`\xi_k` such that :math:`H_k = \{i: \tau_i = \xi_k, \delta_i = 1\}`, and let :math:`m_k`
#         be the number of subjects that have an event at time :math:`\xi_k` such that :math:`m_k = |H_k|`.

#         .. math::

#             pll = \sum_{k} \left( {\sum_{i\in H_{k}}\log \theta_i} - m_k \: \log\left(\sum_{j \in R(\tau_k)} \theta_j \right) \right)


#         **Ties in event time handled with Efron's method.**
#         An alternative approach that is considered to give better results is the Efron's method :cite:p:`Efron1977`.
#         As a compromise between the Cox's and Breslow's method, Efron suggested to use the average
#         risk among the subjects that have an event at time :math:`\xi_k`:

#         .. math::

#             \bar{\theta}_{k} = {\frac {1}{m_{k}}}\sum_{i\in H_{k}}\theta_i

#         Efron approximation of the partial log likelihood is defined by

#         .. math::

#             pll = \sum_{k} \left( {\sum_{i\in H_{k}}\log \theta_i} - \sum_{r =0}^{m_{k}-1} \log\left(\sum_{j \in R(\xi_k)}\theta_j-r\:\bar{\theta}_{j}\right)\right)


#     Examples:
#         >>> log_hz = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
#         >>> event = torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool)
#         >>> time = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
#         >>> neg_partial_log_likelihood(log_hz, event, time)  # default, mean of log likelihoods across patients
#         tensor(1.0071)
#         >>> neg_partial_log_likelihood(log_hz, event, time, reduction="sum")  # sum of log likelihoods across patients
#         tensor(3.0214)
#         >>> time = torch.tensor([1.0, 2.0, 2.0, 4.0, 5.0])  # Dealing with ties (default: Efron)
#         >>> neg_partial_log_likelihood(log_hz, event, time, ties_method="efron")
#         tensor(1.0873)
#         >>> neg_partial_log_likelihood(log_hz, event, time, ties_method="breslow")  # Dealing with ties (Breslow)
#         tensor(1.0873)

#     References:

#         .. bibliography::
#             :filter: False

#             Cox1972
#             Breslow1975
#             Efron1977

#     """  # noqa: E501

#     # if checks:
#     #     validate_survival_data(event, time)
#     #     validate_model(log_hz, event, model_type="cox")

#     if any([event.sum().item() == 0, len(log_hz.size()) == 0]):
#         warnings.warn(
#             "No events OR single sample. Returning zero loss for the batch",
#             stacklevel=2,
#         )
#         return torch.tensor(0.0, requires_grad=True)

#     # sort data by event or censoring time
#     time_sorted, idx = torch.sort(time)
#     log_hz_sorted = log_hz[idx]
#     event_sorted = event[idx]
#     time_unique = torch.unique(time_sorted)  # event or censoring time without ties

#     if len(time_unique) == len(time_sorted):
#         # if not ties, use traditional cox partial likelihood
#         pll = _partial_likelihood_cox(log_hz_sorted, event_sorted)
#     else:
#         # add warning about ties
#         warnings.warn(
#             f"Ties in `time` detected; using {ties_method}'s method to handle ties.",
#             stacklevel=2,
#         )
#         # if ties, use either efron or breslow approximation of partial likelihood
#         if ties_method == "efron":
#             pll = _partial_likelihood_efron(
#                 log_hz_sorted,
#                 event_sorted,
#                 time_sorted,
#                 time_unique,
#             )
#         elif ties_method == "breslow":
#             pll = _partial_likelihood_breslow(log_hz_sorted, event_sorted, time_sorted)
#         else:
#             raise ValueError(f'Ties method {ties_method} should be one of ["efron", "breslow"]')

#     # Negative partial log likelihood
#     pll = torch.neg(pll)

#     #     # Free memory
#     # del tb, eb, log_haz, sindex, log_hazdenom, plls
#     if reduction.lower() == "mean":
#         loss = pll.nanmean()
#     elif reduction.lower() == "sum":
#         loss = pll.sum()
#     else:
#         raise (ValueError(f"Reduction {reduction} is not implemented yet, should be one of ['mean', 'sum']."))
#     return loss


# ### DeepSurv Loss. 
# # Cox models use the Breslow estimator to obtain indiv. risk curves from the baseline hazard of
# # the population at each time point multiplied by the individual log partial hazard (= the model
# # output). For the loss, however, it is sufficient to input the log partial hazard.
# # The loss is explained here https://k-d-w.org/blog/2019/07/survival-analysis-for-deep-learning

# # Adapted from the auton-survival package
# # https://github.com/autonlab/auton-survival/blob/master/auton_survival/models/cph/dcph_utilities.py


def partial_ll_loss(log_haz, tb, eb, eps=1e-3):
    """
    Compute the partial log-likelihood loss.

    Args:
    log_haz: logits representing the log partial hazards (n, )
    tb: time of events in batch (n, )
    eb: event indicator in batch (n, )
    eps: small value for numerical stability"""

    device = log_haz.device

    # Reshape to 1d
    log_haz = log_haz.view(-1)
    tb = tb.view(-1).detach()  # .cpu()
    eb = eb.view(-1).detach()  # .cpu()

    # Remove nans
    idx = torch.isnan(tb).to(device) | torch.isnan(eb).to(device) | torch.isnan(log_haz)
    log_haz = log_haz[~idx]
    tb = tb[~idx]
    eb = eb[~idx]

    # Sort
    tb = tb + eps * torch.rand(len(tb)).to(device)
    sindex = torch.argsort(-tb)
    eb = eb[sindex]
    log_haz = log_haz[sindex]

    if len(log_haz) == 0:
        return torch.tensor(0.0).to(device)

    log_hazdenom = torch.logcumsumexp(log_haz, dim=0)

    plls = log_haz - log_hazdenom
    pll = plls[eb == 1]

    pll = torch.sum(pll)

    # Free memory
    del tb, eb, log_haz, sindex, log_hazdenom, plls

    return -pll.to(device)


class CoxPHLoss(nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, log_partial_hazard, events, durations):
        loss = partial_ll_loss(log_partial_hazard, tb = durations, eb = events)
        return loss


def predict_survival(breslow: BreslowEstimator, preds, times=None):
    """Predict survival function from model and data using Breslow estimator.

    Args:
        breslow: Breslow estimator
            (output of fit_breslow() after training or init_breslow() after loading)
        preds: partial hazard predictions (n, )
        times: time points to include for interpolation (optional)
    """

    if isinstance(times, (int, float)):
        times = [times]

    preds = preds.detach().cpu().numpy()

    with np.errstate(divide="ignore", over="ignore"):
        unique_times = breslow.unique_times_
        # print("Unique times:", unique_times)

        predicted_functions = breslow.get_survival_function(preds)

        arr = np.empty((preds.shape[0], unique_times.shape[0]), dtype=float)
        for i, fn in enumerate(predicted_functions):
            # Get predicted values at the unique event times
            arr[i, :] = fn(unique_times)

        # Get subset of arr at times
        if times is None:
            return arr
        else:
            arr = arr[:, np.isin(unique_times, times)]
            arr[arr > 1] = 1
            arr[arr < 0] = 0
            return arr

def augur_predict_survival(breslow: BreslowEstimator, preds, times=None):
    """Predict survival function from model and data using Breslow estimator.

    Args:
        breslow: Breslow estimator
            (output of fit_breslow() after training or init_breslow() after loading)
        preds: partial hazard predictions (n, )
        times: time points to include for interpolation (optional)
    """

    # if isinstance(times, (int, float)):
    #     times = [times]
    # times = np.asarray(times, dtype = float)

    preds = preds.detach().cpu().numpy()

    with np.errstate(divide="ignore", over="ignore"):
        unique_times = np.asarray(breslow.unique_times_, dtype=float)
        predicted_functions = breslow.get_survival_function(preds)
        # print(predicted_functions)

        # Decide which time grid to use
        if times is None:
            t = unique_times
        else:
            if isinstance(times, (int, float)):
                times = [times]
            t = np.asarray(times, dtype=float)

        # Evaluate survival at t
        arr = np.empty((preds.shape[0], t.shape[0]), dtype=float)
        for i, fn in enumerate(predicted_functions):
            arr[i, :] = fn(t)

        arr[arr > 1] = 1
        arr[arr < 0] = 0

        return arr, t

def predict_cumulative_hazard(breslow: BreslowEstimator, preds, times=None):
    """Predict cumulative hazard function from model and data using Breslow estimator.

    Args:
        breslow: Breslow estimator
            (output of fit_breslow() after training or init_breslow() after loading)
        preds: partial hazard predictions (n, )
        times: time points to include for interpolation (optional)
    """

    if isinstance(times, (int, float)):
        times = [times]

    preds = preds.detach().cpu().numpy()

    with np.errstate(divide="ignore", over="ignore"):
        unique_times = breslow.unique_times_

        predicted_functions = breslow.get_cumulative_hazard_function(preds)

        arr = np.empty((preds.shape[0], unique_times.shape[0]), dtype=float)
        for i, fn in enumerate(predicted_functions):
            # Get predicted values at the unique event times
            arr[i, :] = fn(unique_times)

        # Get subset of arr at times
        if times is None:
            return arr
        else:
            arr = arr[:, np.isin(unique_times, times)]
            return arr
