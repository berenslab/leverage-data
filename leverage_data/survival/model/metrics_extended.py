from datetime import datetime
import json
import sys
import os
from types import SimpleNamespace
from typing import List, Union
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    brier_score_loss,
)
from scipy.interpolate import InterpolatedUnivariateSpline

from sksurv.metrics import (
    brier_score,
    integrated_brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
)
from survival_on_embedding.common.conversion_utils import e_t_to_tuple, et_tuple_to_df
from survival_on_embedding.utils.cnn_survival_utils import get_event_indicator_matrix


# Whether to evaluate on all passed evaluation times or only on all but the last x ones
# TRUNCATE_TIMES_BY = 1

# Fallback method for finding the best threshold if none is specified in the config
DEFAULT_THRESHOLD_FINDER = "rocpr_euclidean"  # "netbenefit_max" #

EXCLUDE_METRICS_CONTAINING = ["buyse"]

try:
    from statkit.decision import net_benefit
except:
    print(
        "Could not import net_benefit from statkit.decision. Will return NaN for net benefit."
    )

try:
    import rpy2

    from rpy2.robjects.packages import importr, isinstalled
    from rpy2.robjects import pandas2ri, numpy2ri, Formula, conversion, default_converter
    # from rpy2.rinterface_lib import openrlib
    numpy2ri.activate()

    packages = ["survival", "timeROC", "BuyseTest"]
    packages = [package for package in packages if not any([exc in package.lower() for exc in EXCLUDE_METRICS_CONTAINING])]

    for package in packages:
        if not isinstalled(package):
            print("installing ", package)
            rutils = importr("utils")
            rutils.chooseCRANmirror(ind=1)
            rutils.install_packages(package)
            print("done")
except:
    print(
        "Could not import timeROC from R. Will return NaN for the respective metrics."
    )


class Metrics:
    """Class for computing metrics for a (survival) CNN model after each epoch.

    Notes:
        - Sksurv metrics with ipcw weighting are computed using the ipcw in the training set.
        - Timeroc metrics are computed using the ipcw in the passed testing data.
        - Net benefit can be computed using statkit and BuyseTest. Statkit is not ipcw-weighted,
          BuyseTest is.
        - Many metrics, including the confusion matrix, are computed at an "optimal" operating
          point as determined by the threshold_finder attribute in the config.
    """

    def __init__(self, config: SimpleNamespace, 
                 target: str = None, 
                 et_train_set: tuple = None,
                 TRUNCATE_TIMES_BY: int = 0):
        self.c = config.cnn
        self.TRUNCATE_TIMES_BY = TRUNCATE_TIMES_BY
        # Load unmodified training data,
        # needed for computation of ipcw in training set for survival metrics
        self.cc = deepcopy(config)
        self.cc.cnn.loss = "any"
        self.cc.cnn.use_stereo_pairs = False
        self.cc.cnn.use_longitudinal_pairs = False

        if et_train_set is None:
            y_train_set = get_dataset(split="train", c=self.cc, target=target).get_e_t()
        else:
            y_train_set = et_train_set

        self.y_train = e_t_to_tuple(
            y_train_set[0],
            y_train_set[1],
        )

        self.reset()
    def safe_float(self, x):
        if x is None or x == rpy2.rinterface.NA_Logical:
            return np.nan
        return float(x)

    def reset(self):
        self.survs = []
        self.events = []
        self.times = []
        self.cumulative_hazards = []

        attributes = [
            "timeroc_tprs",
            "timeroc_fprs",
            "timeroc_ppvs",
            "timeroc_npvs",
            "timeroc_tprs_1000",
            "timeroc_fprs_1000",
            "timeroc_ppvs_1000",
            "timeroc_npvs_1000",
            "timeroc_cutpoints",
            "timeroc_stats",
            "timeroc_ipcw_weights",
            "timeroc_naive_ipcw_weights",
            "pr_timeroc",
            "threshold",
            "threshold_strategy",
            "netbenefits_1000_treated",
            "netbenefits_1000_untreated",
            "net_benefit_cutpoints",
            "buyse",
            "survs_np",
            "times_np",
            "events_np",
            "y_test_np",
            "cumulative_hazards_np",
            "removed_idx",
            "survs_np_cleaned",
            "times_np_cleaned",
            "events_np_cleaned",
            "y_test_np_cleaned",
            "cumulative_hazards_np_cleaned",
            "val_losses",
        ]

        for attr in attributes:
            self._reset_attr(attr)

    def _reset_attr(self, attr):
        if hasattr(self, attr):
            delattr(self, attr)

    def print_surv_info(self, message: str = ""):
        self.add_numpy_preds()

        (
            self.truncated_survival_times,
            self.truncated_survs,
        ) = self.truncate_survtimes_preds()

        print("#### INFO ####")
        print(message)
        for var in [
            "truncated_survival_times",
            "truncated_survs",
            "events_np",
            "y_test_np",
        ]:
            print(f"{var} dtype: {type(getattr(self, var))}")
            print(f"{var} values[0]: {getattr(self, var)[0]}")
            print(f"{var} shape: {getattr(self, var).shape}")

    def truncate_survtimes_preds(self):
        by = self.TRUNCATE_TIMES_BY
        survival_times = np.array(self.c.survival_times)
        # print('survival times', survival_times)
        survs = self.survs_np_cleaned
        if self.TRUNCATE_TIMES_BY > 0:
            survival_times = np.array(self.c.survival_times)[:-by]
            survs = self.survs_np_cleaned[:, :-by]
        return survival_times, survs

    def update(
        self,
        preds: torch.Tensor, # (n, n_times) scalars are floats
        events: torch.Tensor, # (n,) scalars are floats 1.0 or 0.0
        times: torch.Tensor, # (n,) scalars are floats from 0.0 to roughly 20.0 
        cumulative_hazards: torch.Tensor = None, 
    ):
        """
        Updates estimator with current batches predictions and labels.

        Args:
            preds: survival predictions, i.e. probabilities for no event = 1-risk
            events: event indicators
            times: time of events or censoring
            cumulative_hazards: partial hazard predictions, optional
        """
        
        events, times = events.cpu(), times.cpu()

        # Update lists
        self.survs.append(preds.detach())
        self.events.append(events.detach())
        self.times.append(times.detach())

        if cumulative_hazards is not None:
            self.cumulative_hazards.append(cumulative_hazards.detach())

    def update_from_cpu(
        self,
        preds: np.array,
        events: np.array,
        times: np.array,
        cumulative_hazards: np.array = None,
    ):
        """
        Updates estimator with the (current batch) of predictions and labels.

        Args:
            preds: survival predictions, i.e. probabilities for no event = 1-risk
            events: event indicators
            times: time of events or censoring
            cumulative_hazards: partial cum. hazard predictions, optional
        """
        events, times = torch.from_numpy(events), torch.from_numpy(times)

        # Update lists
        self.survs.append(torch.from_numpy(preds))
        self.events.append(events)
        self.times.append(times)

        if cumulative_hazards is not None:
            self.cumulative_hazards.append(torch.from_numpy(cumulative_hazards))

    def add_numpy_preds(self) -> None:
        if not hasattr(self, "survs_np"):
            self.survs_np = torch.cat(self.survs, dim=0).numpy()  # (n, n_times) or (n,)
            if len(self.survs_np.shape) == 1:
                self.survs_np = self.survs_np.reshape(-1, 1)  # (n, ) -> (n, 1)
            self.times_np = (
                torch.cat(self.times, dim=0).numpy().astype(int).squeeze()
            )  # -> (n, 1) -> (n,)
            self.events_np = (
                torch.cat(self.events, dim=0).numpy().astype(int).squeeze()
            )  # -> (n, 1) -> (n,)
            self.y_test_np = e_t_to_tuple(
                self.events_np, self.times_np
            )  # struct. arr (events, times)

            if len(self.cumulative_hazards) > 0 and not hasattr(
                self, "cumulative_hazards_np"
            ):
                self.cumulative_hazards_np = torch.cat(
                    self.cumulative_hazards, dim=0
                ).numpy()  # (n, n_times)
            
            self.add_cleaned_numpy_preds()
                

    def add_cleaned_numpy_preds(self) -> None:
        # Remove rows with negative times. Negative times can be present for the current label
        # in a multilabel setup: The event of interest happened at a later time, with a
        # duration of -x years to the current record.
        idx = np.where(self.times_np < 0)[0]
        self.survs_np_cleaned = np.delete(self.survs_np, idx, axis=0)
        self.times_np_cleaned = np.delete(self.times_np, idx, axis=0)
        self.events_np_cleaned = np.delete(self.events_np, idx, axis=0)

        self.y_test_np_cleaned = e_t_to_tuple(
            self.events_np_cleaned, self.times_np_cleaned
        )  # struct. arr (events, times)

        if hasattr(self, "cumulative_hazards_np"):
            self.cumulative_hazards_np_cleaned = np.delete(
                    self.cumulative_hazards_np, idx, axis=0
                )

        self.removed_idx = idx

    def add_val_loss(self, val_loss: float):
        if hasattr(self, "val_losses"):
            self.val_losses.append(val_loss)
        else:
            self.val_losses = [val_loss]

    def find_threshold(self, method=None):
        """Find the best decision threshold (operating point) for our model using the mean
        performance over all evaluation times. If no method is specified, the threshold_finder
        attribute from the config is used and fallback method is DEFAULT_THRESHOLD_FINDER.
        All choices make use of R's timeroc metrics (ipcw-
        weighted cumulative dynamic approach), except for netbenefit, which uses cumulative-dynamic
        case-control definition but is not ipcw-weighted. The finder can do the following:
        a) maximize sensitivity, specificity, precision, f1, or overall weighted netbenefit -or-
        b) balance the two axes of pr or roc curves:
            i) minimize the difference between them -or-
            ii) minimize the distance to the top-right corner of the PR curve or top-left of
                the ROC curve (euclidean distance)
        c) balance the three axes of roc and pr curves: tpr, fpr, ppv

        Args:
            method (str): {sensitivity, specificity, precision, f1, netbenefit}_{max} or {pr and/or roc}_{mindiff, euclidean}
                examples: "sensitivity_max", "f1_max", "pr_mindiff", "pr_euclidean", "prroc_euclidean"
        Returns:
            threshold (float): Best decision threshold
        """
        # Note: The F1 score does not take into account true negatives (-> part of specificity),
        #   so it may not be suitable for situations where true negatives are important. Conversely,
        #   Youden's J does not take into account the balance between precision and recall, so it may
        #   not be suitable for situations where it's important to maintain a balance between these
        #   two measures. When passing roc_mindiff, this is equivalent to Youden's J.

        if hasattr(self, "threshold"):
            return self.threshold

        self.add_numpy_preds()

        if method is None or method == "":
            method = (
                self.c.threshold_finder
                if hasattr(self.c, "threshold_finder")
                else DEFAULT_THRESHOLD_FINDER
            )

        assert (
            len(method.split("_")) == 2
        ), "Invalid. Pass a method in the form of {metric}_{strategy}"

        strategy = method.split("_")[1]
        metric = method.split("_")[0]

        assert strategy in ["max", "mindiff", "euclidean"], "Invalid strategy"
        assert metric in [
            "sensitivity",
            "recall",
            "specificity",
            "precision",
            "ppv",
            "netbenefit",
            "pr",
            "roc",
            "f1",
            "prroc",
            "rocpr",
        ], "Invalid metric"

        if not hasattr(self, "timeroc_tprs_1000") or self.timeroc_tprs_1000 is None:
            self.calc_SeSpPPVNPV_timeroc(threshold=None)

        # Get rates at 1000 thresholds at each eval_time. shape = (n_times, n_thresholds)
        # and use the mean over all eval_times
        tprs = np.mean(self.timeroc_tprs_1000, axis=0)
        fprs = np.mean(self.timeroc_fprs_1000, axis=0)
        ppvs = np.mean(self.timeroc_ppvs_1000, axis=0)
        cutpoints = self.timeroc_cutpoints

        def _dist_to_corner(x, y, corner=(0, 1)):
            return np.sqrt((x - corner[0]) ** 2 + (y - corner[1]) ** 2)

        def _dist_to_corner3d(x, y, z, corner=(1, 1, 1)):
            return np.sqrt(
                (x - corner[0]) ** 2 + (y - corner[1]) ** 2 + (z - corner[2]) ** 2
            )

        if metric in ["sensitivity", "recall"]:
            if strategy == "max":
                idx = np.argmax(tprs)
            else:
                raise ValueError(f"Invalid strategy for metric {metric}: {strategy}")
        elif metric == "specificity":
            if strategy == "max":
                idx = np.argmin(fprs)
            else:
                raise ValueError(f"Invalid strategy for metric {metric}: {strategy}")
        elif metric in ["precision", "ppv"]:
            if strategy == "max":
                idx = np.argmax(ppvs)
            else:
                raise ValueError(f"Invalid strategy for metric {metric}: {strategy}")
        elif metric == "f1":
            if strategy == "max":
                f1s = 2 / (1 / ppvs + 1 / tprs)
                idx = np.argmax(f1s)
            else:
                raise ValueError(f"Invalid strategy for metric {metric}: {strategy}")
        elif metric == "netbenefit":
            if strategy == "max":
                if not hasattr(self, "netbenefits_1000_treated"):
                    self.calc_netbenefits_statkit(treated=True)
                if not hasattr(self, "netbenefits_1000_untreated"):
                    self.calc_netbenefits_statkit(treated=False)
                nbs_treated = np.mean(self.netbenefits_1000_treated, axis=0)
                nbs_untreated = np.mean(self.netbenefits_1000_untreated, axis=0)
                nbs = nbs_treated + nbs_untreated / 10  # Weighted sum
                idx = np.argmax(nbs)
            else:
                raise ValueError(f"Invalid strategy for metric {metric}: {strategy}")
        elif metric == "pr":
            if strategy == "mindiff":
                idx = np.argmin(np.abs(tprs - ppvs))
            elif strategy == "euclidean":
                idx = np.argmin(_dist_to_corner(tprs, ppvs, (1, 1)))

            else:
                raise ValueError(f"Invalid strategy for metric {metric}: {strategy}")
        elif metric == "roc":
            if strategy == "mindiff":
                idx = np.argmin(np.abs(tprs - (1 - fprs)))
            elif strategy == "euclidean":
                idx = np.argmin(_dist_to_corner(fprs, tprs, (0, 1)))
            else:
                raise ValueError(f"Invalid strategy for metric {metric}: {strategy}")
        elif metric == "prroc" or metric == "rocpr":
            corner = (1, 1, 1)
            v1 = 1 - fprs
            v2 = tprs
            v3 = ppvs
            if strategy == "mindiff":
                idx = np.argmin(np.abs(v1 - v2 - v3))
            elif strategy == "euclidean":
                idx = np.argmin(_dist_to_corner3d(v1, v2, v3, corner))
            else:
                raise ValueError(f"Invalid strategy for metric {metric}: {strategy}")
        else:
            raise ValueError(f"Invalid metric: {metric}")

        self.threshold = cutpoints[idx]
        self.threshold_strategy = method

        return self.threshold

    def find_closest_idx(self, array, value):
        array = np.asarray(array)
        idx = np.argmin(np.abs(array - value))
        return idx

    ########## Calculations ##########

    def calc_event_indicator_matrix(self):
        """Matrix with 1s at all visits >= duration to event (cumulative cases). Times are columns,
        patients are rows. The column index corresponds to time in the unit given by the model's
        durations/times array!

        Example: times = [1, 2, 3, 4, 5]
                 labels are then found through: calc_event_indicator_matrix()[:, times]
        """
        self.add_numpy_preds()

        return get_event_indicator_matrix(
            events=self.events_np_cleaned, durations=self.times_np_cleaned
        )

    def calc_rocauc_curves_sklearn(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()
        event_indicator_matrix = self.calc_event_indicator_matrix()

        tprs = []
        fprs = []
        aucs = []

        for idx, time in enumerate(survival_times):
            fpr, tpr, thres = roc_curve(
                y_true=event_indicator_matrix[:, time],
                y_score=1.0 - survs[:, idx],
            )
            tprs.append(tpr)
            fprs.append(fpr)
            aucs.append(auc(fpr, tpr))

        return tprs, fprs, np.array(aucs)

    def calc_precision_recall_curves_sklearn(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()
        event_indicator_matrix = self.calc_event_indicator_matrix()

        auprcs = []
        precisions = []
        recalls = []

        for idx, time in enumerate(survival_times):
            precision, recall, _ = precision_recall_curve(
                y_true=event_indicator_matrix[:, time],
                y_score=1.0 - survs[:, idx],
            )
            auprc = auc(recall, precision)
            auprcs.append(auprc)
            precisions.append(precision)
            recalls.append(recall)

        return precisions, recalls, np.array(auprcs)

    def calc_SeSpPPVNPV_timeroc(self, threshold: float = None):
        """Computes the comulative-dynamic sensitivity, specificity, positive predictive value and
            negative predictive value from time-varying risk predictions for all evaluation times
            at the passed threshold or over 1000 thresholds. This is censoring adjusted using
            Kaplan-Meier to predict the inverse probability of censoring weights (ipcw). Needs R's 
            timeroc and survival packages installed.

        Args:
            threshold (float): Threshold for binary classification. If float, calculates for this
                threshold only. If None, calculates for 1000 thresholds and sets the follwing
                attributes: timeroc_tprs_1000, timeroc_fprs_1000, timeroc_ppvs_1000,
                timeroc_npvs_1000, timeroc_stats, timeroc_cutpoints, timeroc_ipcw_weights (= the
                probability of not being censored at each eval_time estimated by Kaplan-Meier).

        Returns:
            tprs (np.array (n_times, n_thresholds)): Sensitivity at each threshold for each eval_time
            fprs (np.array (n_times, n_thresholds)): 1 - Specificity at each threshold for each eval_time
            ppvs (np.array (n_times, n_thresholds)): Positive Predictive Value at each threshold for each eval_time
            npvs (np.array (n_times, n_thresholds)): Negative Predictive Value at each threshold for each eval_time
        """
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        with conversion.localconverter(default_converter):
            timeROC = importr("timeROC")
            survival = importr("survival")

        risks = 1.0 - survs

        n_thresholds = 1000
        cuts = np.arange(0, 1, 1 / n_thresholds) if threshold is None else [threshold]

        tprs = []
        fprs = []
        ppvs = []
        npvs = []
        stats = []
        cutpoints = []
        weights = []
        weights_km = []

        for cid, cut in enumerate(cuts):
            tpr_at_time = []
            fpr_at_time = []
            ppv_at_time = []
            npv_at_time = []
            stats_at_time = []

            # SeSpPPVNPV was designed for constant risks. We have to adjust it for our time-
            # dependent risks: For each eval time, we use the corresponding risk column to compute
            # the metrics up to including the eval time and use the last value of the outputs.
            for tid, time in enumerate(survival_times):
                with conversion.localconverter(default_converter):
                    numpy2ri.activate()
                    
                    se_sp_ppv_npv = timeROC.SeSpPPVNPV(
                        cutpoint=cut,
                        T=self.times_np_cleaned,
                        delta=self.events_np_cleaned,
                        marker=risks[:, tid],
                        cause=1,
                        weighting="marginal",  # KM-IPCW
                        times=survival_times[: tid + 1],
                    )
                    numpy2ri.deactivate()

                    # Get the last values of the Se, Sp, PPV, NPV as these correspond to the
                    # cumulated data up to the eval time for the risk column at the eval time.
                    tpr_at_time.append(self.safe_float(se_sp_ppv_npv.rx2("TP")[-1]))
                    fpr_at_time.append(self.safe_float(se_sp_ppv_npv.rx2("FP")[-1]))
                    ppv_at_time.append(self.safe_float(se_sp_ppv_npv.rx2("PPV")[-1]))
                    npv_at_time.append(self.safe_float(se_sp_ppv_npv.rx2("NPV")[-1]))

                    stats_matrix = np.array(se_sp_ppv_npv.rx2("Stats"))
                    row_names = list(se_sp_ppv_npv.rx2("Stats").rownames)
                    col_names = list(se_sp_ppv_npv.rx2("Stats").colnames)
                    stats_df = pd.DataFrame(
                        stats_matrix, columns=col_names, index=row_names
                    )
                    stats_at_time.append(stats_df.iloc[-1])

                    if cid == 0:
                        # Get the naive ipcw weights from the statistics
                        n_cumulative_cases_until_time = stats_at_time[-1]["Cases"]
                        n_survivor_at_time = stats_at_time[-1]["survivor at t"]
                        n_censored_at_time = stats_at_time[-1]["Censored at t"]
                        n_controls_at_time = n_survivor_at_time + n_censored_at_time
                        cum_n = n_controls_at_time + n_cumulative_cases_until_time
                        weights.append(1.0 - (n_censored_at_time / cum_n))

                        # Get the kaplan-meier estimate of the ipcw
                        weights_km.append(
                            np.array(se_sp_ppv_npv.rx2("weights").rx2("IPCW.times"))[-1]
                        )

            tprs.append(tpr_at_time)
            fprs.append(fpr_at_time)
            ppvs.append(ppv_at_time)
            npvs.append(npv_at_time)
            cutpoints.append(cut)
            stats.append(pd.DataFrame(stats_at_time))

        # To (n_times, n_thresholds)
        tprs = np.array(tprs).T
        fprs = np.array(fprs).T
        ppvs = np.array(ppvs).T
        npvs = np.array(npvs).T

        if threshold is not None:
            self.timeroc_tprs = tprs  # (n_times, )
            self.timeroc_fprs = fprs  # (n_times, )
            self.timeroc_ppvs = ppvs  # (n_times, )
            self.timeroc_npvs = npvs  # (n_times, )
            self.timeroc_naive_ipcw_weights = weights  # (n_times, )
            self.timeroc_ipcw_weights = weights_km  # (n_times, )
        else:
            self.timeroc_tprs_1000 = tprs  # (n_times, n_thresholds)
            self.timeroc_fprs_1000 = fprs  # (n_times, n_thresholds)
            self.timeroc_ppvs_1000 = ppvs  # (n_times, n_thresholds)
            self.timeroc_npvs_1000 = npvs  # (n_times, n_thresholds)
            self.timeroc_stats = stats  # (n_thresholds, n_times)
            self.timeroc_cutpoints = cutpoints  # (n_thresholds, )
            self.timeroc_naive_ipcw_weights = weights  # (n_times, )
            self.timeroc_ipcw_weights = weights_km

        return tprs, fprs, ppvs, npvs

    def calc_confmats_timeroc(self, threshold=None, method=None, weighted=True):
        """Read out best threshold from find_threshold() and calculate the confusion matrix for
            each eval_time using ipcw-weighting from timeroc.

        Args:
            threshold (float): Threshold for binary classification.
            method (str): Method to find the best threshold.
                If both are None, the method specified in the threshold_finder attribute from the
                config is used to determine the optimal operating point.
            cumulative_dynamic (bool): If True, uses the cumulative dynamic approach to calculate
                the confusion matrix. If False, uses the classical approach which is not summing
                up the cases up to the inquired evalualtion time.

        Returns:
            confmats (List[np.array]): List of confusion matrices at the threshold for each eval_time:
                [confmat_0, confmat_1, ..., confmat_n_times]. Each confmat is a 2x2 np.array with
                (TN, FP), (FN, TP) as entries; x-axis is the predicted value, y-axis is the actual,
                column and row names are (0,1).
        """

        self.add_numpy_preds()
        if not hasattr(self, "timeroc_tprs_1000") or self.timeroc_tprs_1000 is None:
            self.calc_SeSpPPVNPV_timeroc(threshold=None)
        if threshold is None:
            threshold = self.find_threshold(method=method)

        # Get fractions at 1000 thresholds at each eval_time. shape = (n_times, n_thresholds)
        tprs = self.timeroc_tprs_1000
        fprs = self.timeroc_fprs_1000
        cutpoints = self.timeroc_cutpoints

        # For each time, get the confusion matrix at the threshold
        survival_times, _ = self.truncate_survtimes_preds()
        confmats = []
        for tid, time in enumerate(survival_times):

            idx = self.find_closest_idx(cutpoints, threshold)

            n_cumulative_cases_until_time = self.timeroc_stats[idx].iloc[tid]["Cases"]
            n_cases_before_time = (
                self.timeroc_stats[idx].iloc[tid - 1]["Cases"] if tid > 0 else 0
            )
            n_cases_at_time = (
                self.timeroc_stats[idx].iloc[tid]["Cases"] - n_cases_before_time
            )
            n_survivor_at_time = self.timeroc_stats[idx].iloc[tid]["survivor at t"]
            n_censored_at_time = self.timeroc_stats[idx].iloc[tid]["Censored at t"]
            n_controls_at_time = n_survivor_at_time + n_censored_at_time
            n_predicted_negatives_unweighted = self.timeroc_stats[idx].iloc[tid][
                "Negative (X<=c)"
            ]
            n_predicted_positives_unweighted = self.timeroc_stats[idx].iloc[tid][
                "Positive (X>c)"
            ]

            n_dynamic_controls_at_time = n_survivor_at_time + n_censored_at_time
            n_at_t = n_controls_at_time + n_cases_at_time
            n_with_cumulated_cases_until_t = (
                n_controls_at_time + n_cumulative_cases_until_time
            )

            if weighted:
                n = n_with_cumulated_cases_until_t
                n_pos = n_cumulative_cases_until_time
                n_neg = n_dynamic_controls_at_time

                # Recover the confusion matrix from the cum-dym SeSpPPVNPV
                tp = int(round(tprs[tid, idx] * n_pos, 0))
                fp = int(round(fprs[tid, idx] * n_neg, 0))
                fn = int(round((1 - tprs[tid, idx]) * n_pos, 0))
                tn = int(round((1 - fprs[tid, idx]) * n_neg, 0))
            else:
                raise NotImplementedError(
                    "Not enough information from timeroc stats to calculate the confusion matrix in the unweighted way."
                )
                # Recover the confusion matrix from the unweighted predicted negatives and positives
                # But still use cumulative cases as they remain cases once converted
                n = n_with_cumulated_cases_until_t
                pp = n_predicted_positives_unweighted
                pn = n_predicted_negatives_unweighted
                ap = n_cumulative_cases_until_time
                an = n - ap
                # fp = ?
                tp = pp - fp
                fn = ap - tp
                tn = an - fn

            confmat = np.array([[tn, fp], [fn, tp]])

            confmats.append(confmat)

        return confmats

    def calc_rocs_timeroc(self):
        if not hasattr(self, "timeroc_tprs_1000") or self.timeroc_tprs_1000 is None:
            self.calc_SeSpPPVNPV_timeroc(threshold=None)  # (n_times, n_thresholds)

        tprs_ = self.timeroc_tprs_1000
        fprs_ = self.timeroc_fprs_1000

        n_times = np.array(tprs_).shape[0]
        assert n_times == len(
            self.truncate_survtimes_preds()[0]
        ), "Number of times mismatch"

        return tprs_, fprs_

    def calc_precision_recall_curves_timeroc(self):
        """Calculate precision and recall over n_thresholds for each eval_time. Uses the
        model's survival predictions at the evaluation time! Is censoring adjusted using Kaplan-Meier
        to predict the censoring distribution. Needs R installed and uses packages
        timeROC and survival.
        """
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        # Get tprs, fprs, ppvs, npvs
        if hasattr(self, "timeroc_tprs_1000") and self.timeroc_tprs_1000 is not None:
            print('in here 1000')
            tprs = self.timeroc_tprs_1000
            fprs = self.timeroc_fprs_1000
            ppvs = self.timeroc_ppvs_1000
            npvs = self.timeroc_npvs_1000
        else:
            print('in hier 19')
            tprs, fprs, ppvs, npvs = self.calc_SeSpPPVNPV_timeroc(threshold=None)

        recalls_ = np.array(tprs).astype(float)
        precisions_ = np.array(ppvs).astype(float)
        print('recalls_',recalls_)
        print('precisions_', precisions_)

        auprcs = []
        precisions = []
        recalls = []
        for tid, time in enumerate(survival_times):
            recall_ = recalls_[tid, :]
            precision_ = precisions_[tid, :]
            precision_[np.isnan(precision_)] = 0

            # Ensure that curve values span the whole range of x axis for AUC calculation
            recall_ = np.append(recall_, 0)
            recall_ = np.insert(recall_, 0, values=1)
            precision_ = np.insert(precision_, 0, values=0)
            precision_ = np.append(precision_, 1)

            precisions.append(precision_)
            recalls.append(recall_)

            auprc = auc(recall_, precision_)
            auprcs.append(auprc)

        self.pr_timeroc = (precisions, recalls, np.array(auprcs))

        return precisions, recalls, np.array(auprcs)

    def calc_auprcs(self, method="timeroc"):
        """Get area under the precision-recall curve.

        Args:
            method (str): If "timeroc", uses Rs timeROC package and computes the AUPRC for each
                evaluation time using ipcw-adjusted sensitivity and precision estimates. It makes
                use of the model's survival predictions at the eval_time.
                If "sklearn", uses sklearn's precision_recall_curve function to compute the AUPRC,
                i.e. does not adjust for censoring. Also uses the model's survival predictions at
                the eval_time.

        Returns:
            auprcs (np.array): AUPRC at each evaluation time
        """
        if method == "timeroc":
            if not "rpy2" in sys.modules:
                print('here in timeroc rpy2')
                return np.nan

            if hasattr(self, "pr_timeroc"):
                print('here in pr_timeroc rpy2')
                return self.pr_timeroc[2]
            else:
                return self.calc_precision_recall_curves_timeroc()[2]
        elif method == "sklearn":
            return self.calc_precision_recall_curves_sklearn()[2]
        else:
            raise ValueError(
                f"Invalid method: {method}. Valid methods are: 'timeroc', 'sklearn'"
            )

    def calc_mean_auprc(self, method="timeroc"):
        if method == "timeroc":
            if not "rpy2" in sys.modules:
                print('hier1')
                return np.nan

            if hasattr(self, "pr_timeroc"):
                print('hier 2')
                return np.mean(self.pr_timeroc[2])
            else:
                
                return np.mean(self.calc_precision_recall_curves_timeroc()[2])
        elif method == "sklearn":
            return np.mean(self.calc_precision_recall_curves_sklearn()[2])
        else:
            return np.nan

    def calc_netbenefits_statkit(self, treated=True, clip: bool = False):
        if not "statkit" in sys.modules:
            return np.nan

        survival_times, survs = self.truncate_survtimes_preds()
        y_ = self.calc_event_indicator_matrix()  # w/ cumulative cases
        risks = 1.0 - np.array(survs)

        net_benefits = []

        for tid, visit in enumerate(survival_times):
            threshs, nbs_at_time = net_benefit(
                y_[:, visit],
                risks[:, tid],
                thresholds=np.arange(0, 1, 1 / 1000),
                action=treated,
            )
            if clip:
                nbs_at_time = np.clip(nbs_at_time, 0, 1)
            net_benefits.append(nbs_at_time)

        net_benefits = np.array(net_benefits)  # (n_times, n_thresholds)

        self.net_benefit_cutpoints = threshs

        if treated:
            self.netbenefits_1000_treated = net_benefits
            return self.netbenefits_1000_treated
        else:
            self.netbenefits_1000_untreated = net_benefits
            return self.netbenefits_1000_untreated

    def calc_netbenefits_at_threshold_statkit(self, treated=True, clip: bool = False):
        if not "statkit" in sys.modules:
            return np.nan
        version = "treated" if treated else "untreated"

        # Threshold the net benefits at the optimal operating point
        if not hasattr(self, "threshold") or self.threshold is None:
            self.find_threshold()

        if not hasattr(self, "netbenefits_1000" + "_" + version):
            self.calc_netbenefits_statkit(treated=treated, clip=clip)
        idx = self.find_closest_idx(self.net_benefit_cutpoints, self.threshold)

        nbfs = (
            self.netbenefits_1000_treated
            if treated
            else self.netbenefits_1000_untreated
        )
        return nbfs[:, idx]

    def calc_aunbcs_statkit(self, treated=True, clip: bool = False):
        if not "statkit" in sys.modules:
            return np.nan
        version = "treated" if treated else "untreated"

        if not hasattr(self, "netbenefits_1000" + "_" + version):
            self.calc_netbenefits_statkit(treated=treated, clip=clip)

        nbfs = (
            self.netbenefits_1000_treated
            if treated
            else self.netbenefits_1000_untreated
        )

        aucs = []
        for tid in range(nbfs.shape[0]):
            threshs = self.net_benefit_cutpoints
            nbs_at_time = nbfs[tid]
            aucs.append(auc(threshs, nbs_at_time))
        return np.array(aucs)

    def calc_netbenefits_winratios_at_threshold_buysetest(
        self, threshold=0.5, treated=True
    ):
        """Calculate net benefits and win ratios at a given threshold using the BuyseTest R package
        with Peron's scoring rule and ipcw correction.

        Args:
            threshold (float): Decision threshold for binary classification (operating point).
            treated (bool): If True, handles every sample with risk prediction >= threshold as
                a member of the treatment group. If False, calculates for untreated."""
        # https://cran.r-universe.dev/BuyseTest/doc/manual.html#S4BuyseTest-class
        if not "rpy2" in sys.modules:
            return np.nan

        if treated is False:
            raise NotImplementedError("Untreated metric not implemented yet.")

        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()
        risks = 1.0 - survs

        _y = et_tuple_to_df(self.y_test_np_cleaned)
        _events = _y["event"].astype(int)
        _durations = _y["time"]
        # Avoid zero durations
        # _durations = np.where(_durations == 0, _durations + .001, _durations) # for peron
        idx_converted = np.where(_durations == 0)
        _durations = np.delete(_durations, idx_converted)
        _events = np.delete(_events, idx_converted)
        risks = np.delete(risks, idx_converted, axis=0)

        with conversion.localconverter(default_converter):
            buyseTest = importr("BuyseTest")

        netbenefits = []
        winratios = []

        for tid, time in enumerate(survival_times):
            netbenefit_at_time = []
            winratio_at_time = []

            risks_at_time = risks[:, tid]
            treat = np.where(risks_at_time >= threshold, 1, 0)

            # Treatment needs 2 unique values
            if len(np.unique(treat)) == 1:
                netbenefits.append(np.nan)
                winratios.append(np.nan)
                continue

            data = pd.DataFrame(
                {
                    "eventtime": _durations,
                    "status": _events,
                    "treatment": (
                        treat if treated else 1 - treat
                    ),  # TODO implement untreated
                }
            )

            # store data to check in external R
            # data.to_csv(os.path.join(get_project_dir(), "r_buyse", f"data_{tid}.csv"), index=False)

            # with openrlib.rlock:
            with conversion.localconverter(default_converter):
                pandas2ri.activate()
                formula = Formula("treatment ~ tte(eventtime, status, threshold=0.01)")
                rs4object = buyseTest.BuyseTest(
                    data=data,
                    formula=formula,
                    keep_pairScore=True,
                    trace=0,
                    scoring_rule="Peron",
                    method_inference="none",
                    correction_uninf=2,  # ipcw
                )
                pandas2ri.deactivate()

                col_names = [
                    "favorable",
                    "unfavorable",
                    "neutral",
                    "uninf",
                    "netBenefit",
                    "winRatio",
                ]
                results = pd.DataFrame(
                    np.array(rs4object.slots["Delta"]), columns=col_names
                )
                netbenefit_at_time.append(results["netBenefit"].values[0])
                winratio_at_time.append(results["winRatio"].values[0])

            netbenefits += netbenefit_at_time
            winratios += winratio_at_time

        if not hasattr(self, "buyse"):
            self.buyse = {
                "treated": {"netbenefits": None, "winratios": None},
                "untreated": {"netbenefits": None, "winratios": None},
            }

        treatment = "treated" if treated else "untreated"
        self.buyse[treatment]["netbenefits"] = netbenefits
        self.buyse[treatment]["winratios"] = winratios

        return netbenefits, winratios

    ########## Getters ##########
    ### Threshold independent metrics ###
    def get_loss(self):
        if not hasattr(self, "val_losses"):
            return np.nan
        return self.val_losses[-1]

    def get_losses(self):
        if not hasattr(self, "val_losses"):
            return np.nan
        return self.val_losses

    def get_hazard_ratio(self):
        # https://stackoverflow.com/questions/77241513/how-to-get-the-probability-density-function-from-coxphsurvivalanalysis-in-scikit

        self.add_numpy_preds()
        survival_times, _ = self.truncate_survtimes_preds()

        if (
            not hasattr(self, "cumulative_hazards_np_cleaned")
            or len(self.cumulative_hazards_np_cleaned) == 0
        ):
            return np.nan

        if len(survival_times) == 1:
            return np.nan

        idx_converters = np.where(self.events_np_cleaned == 1)[0]
        idx_nonconverters = np.where(self.events_np_cleaned == 0)[0]

        cumhaz_converters = self.cumulative_hazards_np_cleaned[
            idx_converters, :
        ]  # (n_converters, n_survival_times)
        cumhaz_nonconverters = self.cumulative_hazards_np_cleaned[
            idx_nonconverters, :
        ]  # (n_nonconverters, n_survival_times

        x = list(range(0, cumhaz_converters.shape[1]))

        chf_spline_c = InterpolatedUnivariateSpline(x, cumhaz_converters.mean(axis=0))
        chf_spline_nc = InterpolatedUnivariateSpline(
            x, cumhaz_nonconverters.mean(axis=0)
        )

        hf_c = chf_spline_c.derivative()(x)
        hf_nc = chf_spline_nc.derivative()(x)

        hazard_ratio = hf_c / hf_nc

        if isinstance(hazard_ratio, (list, np.ndarray)):
            # assert np.allclose(hazard_ratio, hazard_ratio[0])
            if not np.allclose(hazard_ratio, hazard_ratio[0]):
                print(
                    "Returning nan; Hazard ratio is not constant over time: ", hazard_ratio
                    )
                return np.nan
            hazard_ratio = hazard_ratio[0]

        return hazard_ratio

    def get_netbenefits_treated_buyse(self):
        if not "rpy2" in sys.modules:
            return np.nan

        if not hasattr(self, "buyse") or self.buyse["treated"]["netbenefits"] is None:
            self.calc_netbenefits_winratios_at_threshold_buysetest(
                threshold=self.find_threshold(), treated=True
            )
        return self.buyse["treated"]["netbenefits"]

    def get_winratios_treated_buyse(self):
        if not "rpy2" in sys.modules:
            return np.nan

        if not hasattr(self, "buyse") or self.buyse["treated"]["winratios"] is None:
            self.calc_netbenefits_winratios_at_threshold_buysetest(
                threshold=self.find_threshold(), treated=True
            )
        return self.buyse["treated"]["winratios"]

    def get_mean_netbenefit_treated_buyse(self):
        netbenefits = self.get_netbenefits_treated_buyse()
        return np.mean(netbenefits)

    def get_mean_winratio_treated_buyse(self):
        winratios = self.get_winratios_treated_buyse()
        return np.mean(winratios)

    def get_aunbcs_treated_statkit(self, clip: bool = False):
        if not "statkit" in sys.modules:
            return np.nan

        aucs = self.calc_aunbcs_statkit(treated=True, clip=clip)
        return aucs

    def get_aunbcs_untreated_statkit(self, clip: bool = False):
        if not "statkit" in sys.modules:
            return np.nan

        aucs = self.calc_aunbcs_statkit(treated=False, clip=clip)
        return aucs

    def get_overall_aunbcs_statkit(self, clip: bool = False):
        aucs = self.get_aunbcs_treated_statkit(
            clip=clip
        ) + self.get_aunbcs_untreated_statkit(clip=clip)
        return np.mean(aucs)

    def get_mean_aunbc_treated_statkit(self, clip: bool = False):
        aucs = self.get_aunbcs_treated_statkit(clip=clip)
        return np.mean(aucs)

    def get_mean_aunbc_untreated_statkit(self, clip: bool = False):
        aucs = self.get_aunbcs_untreated_statkit(clip=clip)
        return np.mean(aucs)

    def get_mean_overall_aunbc_statkit(self, clip: bool = False):
        aucs = self.get_aunbcs_treated_statkit(
            clip=clip
        ) + self.get_aunbcs_untreated_statkit(clip=clip)
        return np.mean(aucs)

    def get_ibs_sksurv(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        # Unlike other sksurv metrics, (I)BS takes survs as input instead of risks:
        # "the estimated probability of remaining event-free up to the i-th time point."
        if len(survival_times) > 1:
            ibs = integrated_brier_score(
                self.y_train, self.y_test_np_cleaned, survs, survival_times
            )
        else:
            ibs = brier_score(self.y_train, self.y_test_np_cleaned, survs, survival_times)[1][0]

        return ibs

    def get_ibs(self):
        """Fall back to get_ibs_sksurv() if user did only specify "ibs" but not the package
        as the model selection metric."""
        return self.get_ibs_sksurv()

    def get_ibs_sklearn(self):
        scores = self.get_brier_scores_sklearn()
        return np.mean(scores)

    def get_aurocs_sksurv(self):
        """The probability that a randomly chosen case (event by t) has a higher risk score than a
          randomly chosen control (event after t). Uses training data to estimate ipc. Does not
          account for zero ipc values, will throw an error if any are present.
           Uses cumulative/dynamic AUC."""
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        try:
            risks = 1.0 - np.array(survs)
            dyn_auc = cumulative_dynamic_auc(
                self.y_train,
                self.y_test_np_cleaned,
                risks,
                survival_times,
            )[0]
        except ValueError as e:
            print(f"Error computing AUC: {e}")
            dyn_auc = np.nan

        return dyn_auc

    def get_aurocs_sklearn(self):
        """The probability that a randomly chosen case (event by t) has a higher risk score than a
          randomly chosen control (event after t). Uses sklearn's AUC w/o ipc weighting and
          w/o cumulative/dynamic approach."""
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        event_indicator_matrix = self.calc_event_indicator_matrix()
        aucs = []
        for idx, time in enumerate(survival_times):
            aucs.append(
                roc_auc_score(
                    y_true=event_indicator_matrix[:, time],
                    y_score=1.0 - survs[:, idx],
                )
            )

        return np.array(aucs)

    def get_mean_auroc_sksurv(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        try:
            risks = 1.0 - np.array(survs)
            mean_auc = cumulative_dynamic_auc(
                self.y_train,
                self.y_test_np_cleaned,
                risks,
                survival_times,
            )[1]
        except ValueError as e:
            print(f"Error computing Mean AUC: {e}")
            mean_auc = np.nan

        return mean_auc

    def get_aurocs_timeroc(self):
        """The probability that a randomly chosen case (event by t) has a higher risk score than a
          randomly chosen control (event after t). Uses the passed data to estimate ipc 
          (testing data). Uses cumulative/dynamic AUC."""
        if not "rpy2" in sys.modules:
            return np.nan

        tprs, fprs = self.calc_rocs_timeroc()

        aucs = []
        for tpr, fpr in zip(tprs, fprs):
            aucs.append(auc(fpr, tpr))

        return np.array(aucs)

    def get_mean_auroc_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan

        aucs = self.get_aurocs_timeroc()
        return np.mean(aucs)

    def get_mean_auroc_sklearn(self):
        aucs = self.get_aurocs_sklearn()
        return np.mean(aucs)

    def get_auprcs_sklearn(self):
        return self.calc_auprcs(method="sklearn")

    def get_mean_auprc_sklearn(self):
        return self.calc_mean_auprc(method="sklearn")

    def get_auprcs_timeroc(self):
        return self.calc_auprcs(method="timeroc")

    def get_mean_auprc_timeroc(self):
        return self.calc_mean_auprc(method="timeroc")

    def get_brier_scores_sksurv(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        # Note: Differently from other sksurv methods, BS takes survs as input:
        # "the estimated probability of remaining event-free up to the i-th time point."
        return brier_score(self.y_train, self.y_test_np_cleaned, survs, survival_times)[1]

    def get_brier_scores_sklearn(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        event_indicator_matrix = self.calc_event_indicator_matrix()
        scores = []
        for idx, time in enumerate(survival_times):
            scores.append(
                brier_score_loss(
                    y_true=event_indicator_matrix[:, time],
                    y_proba=1.0 - survs[:, idx],
                )
            )

        return np.array(scores)

    def get_youden_indices_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan

        if not hasattr(self, "timeroc_tprs_1000") or self.timeroc_tprs_1000 is None:
            self.calc_SeSpPPVNPV_timeroc(threshold=None)

        tprs = self.timeroc_tprs_1000
        fprs = self.timeroc_fprs_1000

        youdens = []
        for tid, time in enumerate(self.truncate_survtimes_preds()[0]):
            y = np.abs(tprs[tid] + (1 - fprs[tid]) - 1)
            idx = np.argmax(y)
            youdens.append(y[idx])

        return np.array(youdens)

    def get_mean_youden_index(self):
        youdens = self.get_youden_indices_timeroc()
        return np.mean(youdens)

    def get_concordance_index_sksurv(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()
        print(f'\n \n \n ***Survival times for C-index calculation: {survival_times} *** \n ')

        # Harrell's C-index over all times
        # Note: This has serious issues when using non-proportional models as risk curves between
        # patients can cross. As discussed here (https://github.com/havakv/pycox/issues/33)
        # it is suggested to either use Antolini's C instead
        # (https://github.com/havakv/pycox/blob/master/pycox/evaluation/concordance.py),
        # or integrate/sum over all risks over times by patient. I do the latter here.
        # Idea: If our model is any good, a patient with shorter time-to-event will have a
        # higher sum of risks than a patient with higher time-to-event.

        # Get sum of risks over all times for each patient
        risks = 1.0 - np.array(survs)
        sum_risk_c = (
            np.sum(risks, axis=1) if len(survival_times) > 1 else risks.flatten()
        )
        # Sort by durations of patients to evaluate
        _y = et_tuple_to_df(self.y_test_np_cleaned)
        _durations = _y["time"]
        _events = _y["event"]
        idx = pd.Series(_durations).sort_values(ascending=False).index
        durations_c = _durations[idx]
        events_c = _events[idx]
        sum_risk_c = sum_risk_c[idx]
        c_harrell = concordance_index_censored(events_c, durations_c, sum_risk_c)[0]

        return c_harrell

    def get_bce_loss(self):
        """Binary cross entropy loss (neg. log likelihood) at the evaluation times specified in the
        config.

        Using sklearn's log_loss or torch.nn.BCELoss(reduction="none") gives the same results
        for a single time point, i.e. when evaluating (a classification model) at one time point
        only. However, when evalauting at multiple time points, they differ. Sticking to torch
        as it explicitly allows for the input and target shapes."""
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        event_indicator_matrix = self.calc_event_indicator_matrix()
        bce = torch.nn.functional.binary_cross_entropy

        predictions = torch.tensor(1.0 - survs).float()
        targets = torch.tensor(event_indicator_matrix[:, survival_times]).float()

        logloss = bce(
            predictions,
            targets,
        ).item()

        return logloss

    def get_ipcws_km_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan

        if not hasattr(self, "timeroc_tprs_1000") or self.timeroc_tprs_1000 is None:
            self.calc_SeSpPPVNPV_timeroc(
                threshold=0.5
            )  # threshold has no influence on the ipcw

        return self.timeroc_ipcw_weights

    #### The below timeroc metrics use the optimal operating point determined by find_threshold() ####

    def get_confmats_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return self.calc_confmats_timeroc()

    def get_mean_confmat_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        confmats = self.get_confmats_timeroc()
        return np.mean(confmats, axis=0)

    def get_confmats_dfs_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        confmats = self.get_confmats_timeroc()
        l = []
        for idx, confmat in enumerate(confmats):
            l.append(
                pd.DataFrame(
                    confmat,
                    columns=["Actual 0", "Actual 1"],
                    index=["Predicted 0", "Predicted 1"],
                )
            )
        return l

    def get_mean_confmat_df_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return pd.DataFrame(
            self.get_mean_confmat_timeroc(),
            columns=["Actual 0", "Actual 1"],
            index=["Predicted 0", "Predicted 1"],
        )

    def get_threshold(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return self.find_threshold()

    def get_threshold_strategy(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return self.threshold_strategy

    def get_tprs_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        if hasattr(self, "timeroc_tprs"):
            return self.timeroc_tprs
        else:
            return self.calc_SeSpPPVNPV_timeroc(self.find_threshold())[0]

    def get_fprs_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        if hasattr(self, "timeroc_fprs"):
            return self.timeroc_fprs
        else:
            return self.calc_SeSpPPVNPV_timeroc(self.find_threshold())[1]

    def get_specificities_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return 1 - self.get_fprs_timeroc()

    def get_ppvs_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        if hasattr(self, "timeroc_ppvs"):
            return self.timeroc_ppvs
        else:
            return self.calc_SeSpPPVNPV_timeroc(self.find_threshold())[2]

    def get_npvs_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        if hasattr(self, "timeroc_npvs"):
            return self.timeroc_npvs
        else:
            return self.calc_SeSpPPVNPV_timeroc(self.find_threshold())[3]

    def get_mean_tpr_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        if hasattr(self, "timeroc_tprs"):
            return np.mean(self.timeroc_tprs)
        else:
            return np.mean(self.calc_SeSpPPVNPV_timeroc(self.find_threshold())[0])

    def get_mean_fpr_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        if hasattr(self, "timeroc_fprs"):
            return np.mean(self.timeroc_fprs)
        else:
            return np.mean(self.calc_SeSpPPVNPV_timeroc(self.find_threshold())[1])

    def get_mean_specificity_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return np.mean(self.get_specificities_timeroc())

    def get_mean_ppv_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        if hasattr(self, "timeroc_ppvs"):
            return np.mean(self.timeroc_ppvs)
        else:
            return np.mean(self.calc_SeSpPPVNPV_timeroc(self.find_threshold())[2])

    def get_mean_npv_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        if hasattr(self, "timeroc_npvs"):
            return np.mean(self.timeroc_npvs)
        else:
            return np.mean(self.calc_SeSpPPVNPV_timeroc(self.find_threshold())[3])

    def get_balanced_accuracies_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        tprs = self.get_tprs_timeroc()
        fprs = self.get_fprs_timeroc()
        return (tprs + (1 - fprs)) / 2

    def get_mean_balanced_accuracy_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return np.mean(self.get_balanced_accuracies_timeroc())

    def get_f1_scores_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        ppvs = self.get_ppvs_timeroc()
        tprs = self.get_tprs_timeroc()
        f1s = 2 / (1 / ppvs + 1 / tprs)
        return f1s

    def get_mean_f1_score_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return np.mean(self.get_f1_scores_timeroc())

    def get_mccs_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        tprs = self.get_tprs_timeroc()
        fprs = self.get_fprs_timeroc()
        tnrs = 1 - fprs
        fnrs = 1 - tprs
        mccs = (tprs * tnrs - fprs * fnrs) / np.sqrt(
            (tprs + fprs) * (tprs + fnrs) * (tnrs + fprs) * (tnrs + fnrs)
        )
        return mccs

    def get_mean_mcc_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return np.mean(self.get_mccs_timeroc())

    def get_markednesses_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        ppvs = self.get_ppvs_timeroc()
        npvs = self.get_npvs_timeroc()
        markednesses = ppvs + npvs - 1
        return markednesses

    def get_mean_markedness_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return np.mean(self.get_markednesses_timeroc())

    def get_pos_likelihood_ratios_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        tprs = self.get_tprs_timeroc()
        fprs = self.get_fprs_timeroc()
        return tprs / fprs

    def get_neg_likelihood_ratios_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        tprs = self.get_tprs_timeroc()
        fprs = self.get_fprs_timeroc()
        return (1 - tprs) / (1 - fprs)

    def get_mean_pos_likelihood_ratio_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return np.mean(self.get_pos_likelihood_ratios_timeroc())

    def get_mean_neg_likelihood_ratio_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return np.mean(self.get_neg_likelihood_ratios_timeroc())

    def get_diagnostic_odds_ratios_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan

        lr_pos = self.get_pos_likelihood_ratios_timeroc()
        lr_neg = self.get_neg_likelihood_ratios_timeroc()
        return lr_pos / lr_neg

    def get_mean_diagnostic_odds_ratio_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan
        return np.mean(self.get_diagnostic_odds_ratios_timeroc())

    def get_netbenefits_treated_statkit(self, clip: bool = False):
        if not ("statkit" in sys.modules and "rpy2" in sys.modules):
            return np.nan

        net_benefits_at_thresh = self.calc_netbenefits_at_threshold_statkit(
            treated=True, clip=clip
        )
        return net_benefits_at_thresh

    def get_netbenefits_untreated_statkit(self, clip: bool = False):
        if not ("statkit" in sys.modules and "rpy2" in sys.modules):
            return np.nan

        net_benefits_at_thresh = self.calc_netbenefits_at_threshold_statkit(
            treated=False, clip=clip
        )
        return net_benefits_at_thresh

    def get_overall_netbenefits_statkit(self, clip: bool = False):
        if not ("statkit" in sys.modules and "rpy2" in sys.modules):
            return np.nan

        treated = self.get_netbenefits_treated_statkit(clip=clip)
        untreated = self.get_netbenefits_untreated_statkit(clip=clip)
        return treated + untreated

    def get_mean_netbenefit_treated_statkit(self, clip: bool = False):
        if not ("statkit" in sys.modules and "rpy2" in sys.modules):
            return np.nan

        net_benefits_at_thresh = self.get_netbenefits_treated_statkit(clip=clip)
        return np.mean(net_benefits_at_thresh)

    def get_mean_netbenefit_untreated_statkit(self, clip: bool = False):
        if not ("statkit" in sys.modules and "rpy2" in sys.modules):
            return np.nan

        net_benefits_at_thresh = self.get_netbenefits_untreated_statkit(clip=clip)
        return np.mean(net_benefits_at_thresh)

    def get_mean_overall_netbenefit_statkit(self, clip: bool = False):
        if not ("statkit" in sys.modules and "rpy2" in sys.modules):
            return np.nan

        net_benefits_at_thresh = self.get_overall_netbenefits_statkit(clip=clip)
        return np.mean(net_benefits_at_thresh)

    def get_stats_timeroc(self):
        if not "rpy2" in sys.modules:
            return np.nan

        if not hasattr(self, "timeroc_stats") or self.timeroc_stats is None:
            self.calc_SeSpPPVNPV_timeroc(threshold=None)

        idx = self.find_closest_idx(self.timeroc_cutpoints, self.find_threshold())
        return self.timeroc_stats[idx]

    #### End of fixed-threshold timeroc metrics ####

    def get_all_performances(self):
        self.add_numpy_preds()
        print("Set length: ", len(self.y_test_np_cleaned))

        out = {}
        for metric in self.get_valid_metrics():
            out[metric] = self.get_performance(metric)

        return out

    def get_performance(self, metric: str):
        try:
            out = getattr(self, f"get_{metric}")()
        except AttributeError as e:
            print(e)
            raise ValueError(
                f"Invalid metric: {metric}. Valid metrics are: {self.get_valid_metrics()}"
            )

        return out

    def get_valid_metrics(self):
        prefix = "get_"
        valid_metrics = [
            metric[len(prefix) :] for metric in dir(self) if metric.startswith("get_")
        ]
        valid_metrics = [
            metric
            for metric in valid_metrics
            if not "metrics" in metric
            and not "labels" in metric
            and not "performance" in metric
        ]

        # Exclude metrics containing any string in EXCLUDE_METRICS_CONTAINING
        valid_metrics = [
            metric
            for metric in valid_metrics
            if not any([exclude in metric.lower() for exclude in EXCLUDE_METRICS_CONTAINING])
        ]

        return valid_metrics

    ########## Plotting ##########

    def plot_roccurve(
        self, method="timeroc", return_values=False, add_operating_point=True
    ):
        self.reset_plot()
        self.add_numpy_preds()

        survival_times, _ = self.truncate_survtimes_preds()

        if method == "sklearn":
            tprs, fprs, aucs = self.calc_rocauc_curves_sklearn()
        elif method == "timeroc":
            tprs, fprs = self.calc_rocs_timeroc()
            aucs = self.get_aurocs_timeroc()
        else:
            raise ValueError(
                f"Invalid method: {method}. Valid methods are: 'sklearn', 'timeroc'"
            )

        fig, ax = plt.subplots(figsize=(6, 6))
        for idx, visit in enumerate(survival_times):
            lines = plt.plot(
                fprs[idx],
                tprs[idx],
                label=f"Year: {int(visit/2)}, AUC: {aucs[idx]:.3f}",
            )
            color = lines[-1].get_color()
            if add_operating_point and method == "timeroc":
                if not hasattr(self, "timeroc_tprs") or self.timeroc_tprs is None:
                    self.calc_SeSpPPVNPV_timeroc(threshold=None)
                threshold = self.find_threshold()
                tprs_ = tprs[idx]
                fprs_ = fprs[idx]
                plt.scatter(
                    fprs_[self.find_closest_idx(self.timeroc_cutpoints, threshold)],
                    tprs_[self.find_closest_idx(self.timeroc_cutpoints, threshold)],
                    color=color,
                    s=40,
                )

        if add_operating_point and method == "timeroc":
            plt.scatter(-1, -1, color="black", s=40, label="Operating Point")

        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_ylim(0, 1.005)
        ax.set_xlim(0, 1.005)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend(loc="lower left", frameon=False)

        plt.show()

        if return_values:
            return fig, ax, tprs, fprs, aucs
        else:
            return fig, ax

    def plot_prcurve(
        self,
        method="timeroc",
        n_thresholds=1000,
        return_values=False,
        add_operating_point=True,
    ):
        self.reset_plot()
        self.add_numpy_preds()

        survival_times, _ = self.truncate_survtimes_preds()

        if method == "timeroc":
            if not "rpy2" in sys.modules:

                if return_values:
                    return [None] * 5
                return [None] * 2

            if hasattr(self, "pr_timeroc"):
                precisions, recalls, auprcs = self.pr_timeroc
            else:
                precisions, recalls, auprcs = self.calc_precision_recall_curves_timeroc(
                    n_thresholds
                )
        elif method == "sklearn":
            precisions, recalls, auprcs = self.calc_precision_recall_curves_sklearn()
        else:
            raise ValueError(
                f"Invalid method: {method}. Valid methods are: 'timeroc', 'sklearn'"
            )

        fig, ax = plt.subplots(figsize=(6, 6))
        for tid, visit in enumerate(survival_times):
            lines = plt.plot(
                recalls[tid],
                precisions[tid],
                label=f"Year: {int(visit/2)}, AUC: {auprcs[tid]:.3f}",
            )
            color = lines[-1].get_color()
            if add_operating_point and method == "timeroc":
                if not hasattr(self, "timeroc_tprs") or self.timeroc_tprs is None:
                    self.calc_SeSpPPVNPV_timeroc(threshold=None)
                threshold = self.find_threshold()
                tprs = self.timeroc_tprs_1000[tid]
                ppvs = self.timeroc_ppvs_1000[tid]
                plt.scatter(
                    tprs[self.find_closest_idx(self.timeroc_cutpoints, threshold)],
                    ppvs[self.find_closest_idx(self.timeroc_cutpoints, threshold)],
                    color=color,
                    s=40,
                )

        if add_operating_point and method == "timeroc":
            plt.scatter(-1, -1, color="black", s=40, label="Operating Point")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1.005)
        ax.set_xlim(0, 1.005)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend(loc="lower left", frameon=False)

        plt.show()

        if return_values:
            return fig, ax, precisions, recalls, auprcs
        else:
            return fig, ax

    def plot_confmats(self):
        self.reset_plot()
        self.add_numpy_preds()

        if not "rpy2" in sys.modules:
            print("Could not import rpy2. Cannot plot confusion matrices.")
            return None, None

        confmats = self.get_confmats_timeroc()

        n = len(confmats)
        ncols = 5 if n >= 5 else n
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

        # If only one subplot, make axs an array of one subplot
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for i, confmat in enumerate(confmats):
            if len(axs) > 1 and axs.ndim > 1:
                ax = axs[i // ncols, i % ncols]
            else:
                ax = axs[i]
            cax = ax.matshow(confmat, cmap=plt.cm.Blues)
            ax.set_title(f"{i+1}")
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            ax.set_xticks(np.arange(2))
            ax.set_yticks(np.arange(2))
            ax.set_xticklabels(["0", "1"])
            ax.set_yticklabels(["0", "1"])

            # Loop over data dimensions and create text annotations.
            n = np.sum(confmat)
            for i in range(confmat.shape[0]):
                for j in range(confmat.shape[1]):
                    count = int(confmat[i, j])
                    frac = str(round(count / n * 100, 1)) + "%" if n > 0 else "0%"
                    ax.text(
                        j,
                        i,
                        f"{count}\n({frac})",
                        ha="center",
                        va="center",
                        color=(
                            "white" if confmat[i, j] > confmat.max() / 2.0 else "black"
                        ),
                    )

        # Remove unused subplots
        if n % ncols != 0:
            for idx in range(n, nrows * ncols):
                fig.delaxes(axs.flatten()[idx])

        # # Add a colorbar to the figure.
        # fig.colorbar(cax, ax=axs, location='right')

        plt.show()

        return fig, axs

    def plot_mean_confmat(self):
        self.reset_plot()
        self.add_numpy_preds()

        if not "rpy2" in sys.modules:
            print("Could not import rpy2. Cannot plot confusion matrices.")
            return

        mean_confmat = self.get_mean_confmat_timeroc()

        fig, ax = plt.subplots(figsize=(5, 5))
        cax = ax.matshow(mean_confmat, cmap=plt.cm.Blues)
        ax.set_title("Mean")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        # Loop over data dimensions and create text annotations.
        n = np.sum(mean_confmat)
        for i in range(mean_confmat.shape[0]):
            for j in range(mean_confmat.shape[1]):
                count = int(mean_confmat[i, j])
                frac = str(round(count / n * 100, 1)) + "%" if n > 0 else "0%"
                ax.text(
                    j,
                    i,
                    f"{count}\n({frac})",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if mean_confmat[i, j] > mean_confmat.max() / 2.0
                        else "black"
                    ),
                )

        # # Add a colorbar to the figure.
        # fig.colorbar(cax, ax=ax, location='right')

        plt.show()

        return fig, ax

    def save_roccurve(
        self,
        path: str = "",
        name: str = "roccurve",
        method="timeroc",
        save_values=False,
    ):

        if not path.startswith("/"):
            path = os.path.join(get_project_dir(), path)

        fig, ax, tprs, fprs, aucs = self.plot_roccurve(method, return_values=True)

        if fig is None:
            print("Could not plot ROC-curve.")
            return

        os.makedirs(path, exist_ok=True)
        for filetype in ["pdf", "png"]:
            fig.savefig(os.path.join(path, f"{name}_{method}.{filetype}"))

        if save_values:
            # Save values for plot reproduction
            df = pd.DataFrame(tprs)
            df.to_csv(os.path.join(path, f"{name}_{method}_tprs.csv"), index=False)
            df = pd.DataFrame(fprs)
            df.to_csv(os.path.join(path, f"{name}_{method}_fprs.csv"), index=False)

    def save_prcurve(
        self, path: str = "", name: str = "prcurve", method="timeroc", save_values=False
    ):

        if not path.startswith("/"):
            path = os.path.join(get_project_dir(), path)

        fig, ax, precisions, recalls, auprcs = self.plot_prcurve(
            method, return_values=True
        )

        if fig is None:
            print("Could not plot PR-curve.")
            return

        os.makedirs(path, exist_ok=True)
        for filetype in ["pdf", "png"]:
            fig.savefig(os.path.join(path, f"{name}_{method}.{filetype}"))

        if save_values:
            # Save values for plot reproduction
            df = pd.DataFrame(precisions)
            df.to_csv(
                os.path.join(path, f"{name}_{method}_precisions.csv"), index=False
            )
            df = pd.DataFrame(recalls)
            df.to_csv(os.path.join(path, f"{name}_{method}_recalls.csv"), index=False)

    def save_confmats(self, path: str = "", name: str = "confmats"):

        if not path.startswith("/"):
            path = os.path.join(get_project_dir(), path)

        fig, axs = self.plot_confmats()

        if fig is None:
            print("Could not plot confusion matrices.")
            return

        os.makedirs(path, exist_ok=True)
        for filetype in ["pdf", "png"]:
            fig.savefig(os.path.join(path, f"{name}.{filetype}"))

        # Mean confusion matrix
        fig, ax = self.plot_mean_confmat()
        for filetype in ["pdf", "png"]:
            fig.savefig(os.path.join(path, f"{name}_mean.{filetype}"))

    def save_all_performances(self, path: str = "", name: str = ""):
        def _save_df(df, path, name):
            df.to_csv(os.path.join(path, f"{name}.csv"), index=True)

        def _save_array(array, path, name):
            np.savetxt(
                os.path.join(path, f"{name}.csv"), array, delimiter=",", fmt="%.20f"
            )

        if not path.startswith("/"):
            path = os.path.join(get_project_dir(), path)

        out = self.get_all_performances()

        # From the confmats, keep only the dfs
        del out["confmats_timeroc"]
        del out["mean_confmat_timeroc"]

        os.makedirs(path, exist_ok=True)
        for metric, value in out.items():
            if isinstance(value, (np.ndarray, list)):
                if hasattr(value, "shape") and len(value.shape) <= 2:
                    _save_array(np.array(value), path, f"{name}_{metric}")
                else:
                    for idx, v in enumerate(value):
                        if isinstance(v, (np.ndarray, list)):
                            _save_array(np.array(v), path, f"{name}_{metric}_{idx}")
                        elif isinstance(v, (pd.DataFrame)):
                            _save_df(v, path, f"{name}_{metric}_{idx}")
                        else:
                            with open(
                                os.path.join(path, f"{name}_{metric}_{idx}.txt"), "a"
                            ) as f:
                                f.write(str(v) + "\n")
            elif isinstance(value, (int, float, str)):
                with open(os.path.join(path, f"{name}.txt"), "a") as f:
                    f.write(f"{metric}: {value}\n")
            elif isinstance(value, (dict)):
                with open(os.path.join(path, f"{name}_{metric}.txt"), "w") as f:
                    for k, v in value.items():
                        f.write(f"{k}: {v}\n")
            elif isinstance(value, (pd.DataFrame)):
                _save_df(value, path, f"{name}_{metric}")
            else:
                with open(os.path.join(path, f"{name}_{metric}.txt"), "a") as f:
                    f.write(str(value) + "\n")

    def save_everything(self, dir: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir += f"_{timestamp}"
        self.save_all_performances(os.path.join(dir, "performances"))
        if "rpy2" in sys.modules:
            self.save_roccurve(
                os.path.join(dir, "roc"), method="timeroc", save_values=True
            )
            self.save_prcurve(
                os.path.join(dir, "pr"), method="timeroc", save_values=True
            )
        self.save_roccurve(os.path.join(dir, "roc"), method="sklearn", save_values=True)
        self.save_prcurve(os.path.join(dir, "pr"), method="sklearn", save_values=True)
        self.save_confmats(os.path.join(dir, "confmats"))

        print("All metrics and plots saved to ", dir)

    def save_config(self, path: str = "", name: str = "config"):
        if not path.startswith("/"):
            path = os.path.join(get_project_dir(), path)

        config = vars(self.cc)

        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, f"{name}.json")
        if os.path.exists(file):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file = os.path.join(path, f"{name}_{timestamp}.json")

        with open(file, "w") as f:
            json.dump(config, f, indent=4)

    def reset_plot(self):
        try:
            plt.close("all")
        except:
            pass
def get_estimator(config: SimpleNamespace, et_train_set: tuple = None):
    return Metrics(config, et_train_set=et_train_set)
def get_estimators(config: SimpleNamespace, et_train_set: tuple = None):
    # # Handle both prediction_targets and legacy pos_labels
    # if hasattr(config.cnn, "prediction_targets"):
    #     labels = deepcopy(config.cnn.prediction_targets)
    # elif hasattr(config.cnn, "pos_labels"): # Legacy
    #     labels = deepcopy(config.cnn.pos_labels)
    # else:
    #     # Default to late AMD if neither is available
    #     labels = ["late"]
        
    # if all([label in labels for label in ["wet", "dry"]]):
    #     labels += ["late"]

    # if isinstance(labels, str):
    #     if labels == "all":
    #         labels = ["late", "dry", "wet"]
    #     else:
    #         labels = [labels]
    if config.cnn.multilabel:
        labels = AMD_TARGETS["multilabel"]["evaluation_targets"]
    else:
        labels = AMD_TARGETS["single"]["evaluation_targets"]

    ccopy = deepcopy(config)

    estimators = {label: None for label in labels}
    for label in labels:
        estimators[label] = Metrics(ccopy, target=label, et_train_set=et_train_set)

    return estimators