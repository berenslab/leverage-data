import torch
import sys
import numpy as np
import pandas as pd
from survival_on_embedding.utils import e_t_to_tuple 
from sklearn.metrics import auc

try:
    from rpy2.robjects.packages import importr, isinstalled
    from rpy2.robjects import pandas2ri, numpy2ri, Formula, conversion, default_converter
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import FloatVector, IntVector


    # from rpy2.rinterface_lib import openrlib

    packages = ["survival", "timeROC"]
    EXCLUDE_METRICS_CONTAINING = []
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

class RMetrics:
    def __init__(self, eval_times, preds, events, times):
        """

        Args:
            preds: survival predictions, i.e. probabilities for no event = 1-risk
            events: event indicators
            times: time of events or censoring
            cumulative_hazards: partial hazard predictions, optional
        """
        self.survival_predictions = preds
        print('preds0', self.survival_predictions[0])
        self.events = events
        self.times = times
        self.eval_times = eval_times
        print('eval_times',self.eval_times)
  
    def add_numpy_preds(self) -> None:
        if not hasattr(self, "survs_np"):
            # self.survs_np = torch.cat(self.survs, dim=0).numpy()  # (n, n_times) or (n,)
            # if len(self.survs_np.shape) == 1:
            #     self.survs_np = self.survs_np.reshape(-1, 1)  # (n, ) -> (n, 1)
            if self.times.ndim > 1:
                self.times_np = (
                torch.cat(self.times, dim=0).numpy().astype(int).squeeze()
            )  # -> (n, 1) -> (n,)
            else:
                self.times_np = self.times.numpy()
            if self.events.ndim > 1:
                self.events_np = (
                torch.cat(self.events, dim=0).numpy().astype(int).squeeze()
            )  # -> (n, 1) -> (n,)
            else:
                self.events_np = self.events.numpy().astype(int)
            self.y_test_np = e_t_to_tuple(
                self.events_np, self.times_np
            )  # struct. arr (events, times)

            # if len(self.cumulative_hazards) > 0 and not hasattr(
            #     self, "cumulative_hazards_np"
            # ):
            #     self.cumulative_hazards_np = torch.cat(
            #         self.cumulative_hazards, dim=0
            #     ).numpy()  # (n, n_times)
            
    
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
        survs = self.survival_predictions

        # Import R packages
        timeROC = importr("timeROC")
        survival = importr("survival")
        
        print('heer1')
        risks = 1.0 - survs

        n_thresholds = 1000
        cuts = np.arange(0, 1, 1 / n_thresholds) if threshold is None else [threshold]

        T_r = FloatVector(self.times_np.tolist())
        delta_r = IntVector(self.events_np.tolist())

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

            for tid, time in enumerate(self.eval_times):
                marker_r = FloatVector(risks[:, tid].tolist())
                times_r = FloatVector(self.eval_times[: tid + 1].tolist())

                res = timeROC.SeSpPPVNPV(
                    cutpoint=float(cut),
                    T=T_r,
                    delta=delta_r,
                    marker=marker_r,
                    cause=1,
                    weighting="marginal",  # KM-IPCW
                    times=times_r
                )

                TP   = float(res.rx2("TP")[-1])
                FP   = float(res.rx2("FP")[-1])
                PPV  = float(res.rx2("PPV")[-1])
                NPV  = float(res.rx2("NPV")[-1])
                Stats = res.rx2("Stats")

                tpr_at_time.append(TP)
                fpr_at_time.append(FP)
                ppv_at_time.append(PPV)
                npv_at_time.append(NPV)

                # Convert R Stats matrix properly
                stats_mat = np.asarray(Stats)
                row_names = list(Stats.rownames)
                col_names = list(Stats.colnames)

                stats_df = pd.DataFrame(
                    stats_mat,
                    index=row_names,
                    columns=col_names
                )
                stats_at_time.append(stats_df.iloc[-1])

                if cid == 0:
                    n_cumulative_cases_until_time = stats_df.iloc[-1]["Cases"]
                    n_survivor_at_time  = stats_df.iloc[-1]["survivor at t"]
                    n_censored_at_time  = stats_df.iloc[-1]["Censored at t"]
                    n_controls_at_time = n_survivor_at_time + n_censored_at_time
                    cum_n = n_controls_at_time + n_cumulative_cases_until_time
                    weights.append(1.0 - (n_censored_at_time / cum_n))
                    weights_km.append(
                                np.array(res.rx2("weights").rx2("IPCW.times"))[-1]
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
    
    def calc_rocs_timeroc(self):
        if not hasattr(self, "timeroc_tprs_1000") or self.timeroc_tprs_1000 is None:
            self.calc_SeSpPPVNPV_timeroc(threshold=None)  # (n_times, n_thresholds)
 
        tprs_ = self.timeroc_tprs_1000
        print('passed tpts', tprs_)
        fprs_ = self.timeroc_fprs_1000
        print('passed fprs_', fprs_)

 
        n_times = np.array(tprs_).shape[0]
        print('n_times', n_times)
        # assert n_times == len(
        #     self.truncate_survtimes_preds()[0]
        # ), "Number of times mismatch"
 
        return tprs_, fprs_
 
    def calc_precision_recall_curves_timeroc(self):
        """Calculate precision and recall over n_thresholds for each eval_time. Uses the
        model's survival predictions at the evaluation time! Is censoring adjusted 
        using Kaplan-Meier
        to predict the censoring distribution. Needs R installed and uses packages
        timeROC and survival.
        """
        self.add_numpy_preds()
 
        # survival_times, survs = self.truncate_survtimes_preds()
 
        # Get tprs, fprs, ppvs, npvs
        if hasattr(self, "timeroc_tprs_1000") and self.timeroc_tprs_1000 is not None:
            tprs = self.timeroc_tprs_1000
            fprs = self.timeroc_fprs_1000
            ppvs = self.timeroc_ppvs_1000
            npvs = self.timeroc_npvs_1000
        else:
            tprs, fprs, ppvs, npvs = self.calc_SeSpPPVNPV_timeroc(threshold=None)
 
        recalls_ = np.array(tprs)
        precisions_ = np.array(ppvs)
 
        auprcs = []
        precisions = []
        recalls = []
        for tid, time in enumerate(self.eval_times):
            print(tid, time)
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
                return np.nan
 
            if hasattr(self, "pr_timeroc"):
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
                return np.nan
 
            if hasattr(self, "pr_timeroc"):
                return np.mean(self.pr_timeroc[2])
            else:
                return np.mean(self.calc_precision_recall_curves_timeroc()[2])
        elif method == "sklearn":
            return np.mean(self.calc_precision_recall_curves_sklearn()[2])
        else:
            return np.nan
 
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
    
    def get_auprcs_timeroc(self):
            return self.calc_auprcs(method="timeroc")


# import torch
# import sys
# import numpy as np
# import pandas as pd
# from survival_on_embedding.utils import e_t_to_tuple 
# from sklearn.metrics import auc

# try:
#     from rpy2.robjects.packages import importr, isinstalled
#     from rpy2.robjects import pandas2ri, numpy2ri, Formula, conversion, default_converter
#     numpy2ri.activate()
#     # from rpy2.rinterface_lib import openrlib

#     packages = ["survival", "timeROC"]
#     EXCLUDE_METRICS_CONTAINING = []
#     packages = [package for package in packages if not any([exc in package.lower() for exc in EXCLUDE_METRICS_CONTAINING])]

#     for package in packages:
#         if not isinstalled(package):
#             print("installing ", package)
#             rutils = importr("utils")
#             rutils.chooseCRANmirror(ind=1)
#             rutils.install_packages(package)
#             print("done")
# except:
#     print(
#         "Could not import timeROC from R. Will return NaN for the respective metrics."
#     )

# class RMetrics:
#     def __init__(self, eval_times, preds, events, times):
#         """

#         Args:
#             preds: survival predictions, i.e. probabilities for no event = 1-risk
#             events: event indicators
#             times: time of events or censoring
#             cumulative_hazards: partial hazard predictions, optional
#         """
#         self.survival_predictions = preds
#         print('preds0', self.survival_predictions[0])
#         self.events = events
#         self.times = times
#         self.eval_times =  np.array(eval_times, dtype=np.float64) #if not isinstance(eval_times, np.ndarray) else eval_times
#         print('eval_times', self.eval_times.dtype)
  
#     def add_numpy_preds(self) -> None:
#         if not hasattr(self, "survs_np"):
#             # self.survs_np = torch.cat(self.survs, dim=0).numpy()  # (n, n_times) or (n,)
#             # if len(self.survs_np.shape) == 1:
#             #     self.survs_np = self.survs_np.reshape(-1, 1)  # (n, ) -> (n, 1)
#             if self.times.ndim > 1:
#                 self.times_np = (
#                 torch.cat(self.times, dim=0).numpy().astype(int).squeeze()
#             )  # -> (n, 1) -> (n,)
#             else:
#                 self.times_np = self.times.numpy()
#             if self.events.ndim > 1:
#                 self.events_np = (
#                 torch.cat(self.events, dim=0).numpy().astype(int).squeeze()
#             )  # -> (n, 1) -> (n,)
#             else:
#                 self.events_np = self.events.numpy().astype(int)
#             self.y_test_np = e_t_to_tuple(
#                 self.events_np, self.times_np
#             )  # struct. arr (events, times)

#             # if len(self.cumulative_hazards) > 0 and not hasattr(
#             #     self, "cumulative_hazards_np"
#             # ):
#             #     self.cumulative_hazards_np = torch.cat(
#             #         self.cumulative_hazards, dim=0
#             #     ).numpy()  # (n, n_times)
            
    
#     def calc_SeSpPPVNPV_timeroc(self, threshold: float = None):
#             """Computes the comulative-dynamic sensitivity, specificity, positive predictive value and
#                 negative predictive value from time-varying risk predictions for all evaluation times
#                 at the passed threshold or over 1000 thresholds. This is censoring adjusted using
#                 Kaplan-Meier to predict the inverse probability of censoring weights (ipcw). Needs R's
#                 timeroc and survival packages installed.
    
#             Args:
#                 threshold (float): Threshold for binary classification. If float, calculates for this
#                     threshold only. If None, calculates for 1000 thresholds and sets the follwing
#                     attributes: timeroc_tprs_1000, timeroc_fprs_1000, timeroc_ppvs_1000,
#                     timeroc_npvs_1000, timeroc_stats, timeroc_cutpoints, timeroc_ipcw_weights (= the
#                     probability of not being censored at each eval_time estimated by Kaplan-Meier).
    
#             Returns:
#                 tprs (np.array (n_times, n_thresholds)): Sensitivity at each threshold for each eval_time
#                 fprs (np.array (n_times, n_thresholds)): 1 - Specificity at each threshold for each eval_time
#                 ppvs (np.array (n_times, n_thresholds)): Positive Predictive Value at each threshold for each eval_time
#                 npvs (np.array (n_times, n_thresholds)): Negative Predictive Value at each threshold for each eval_time
#             """
#             self.add_numpy_preds()
#             survs = self.survival_predictions
    
#             # survival_times, survs = self.truncate_survtimes_preds()
    
#             # with conversion.localconverter(default_converter):
#             timeROC = importr("timeROC")
#             survival = importr("survival")
            

#             risks = 1.0 - survs
    
#             n_thresholds = 1000
#             cuts = np.arange(0, 1, 1 / n_thresholds) if threshold is None else [threshold]
    
#             tprs = []
#             fprs = []
#             ppvs = []
#             npvs = []
#             stats = []
#             cutpoints = []
#             weights = []
#             weights_km = []
#             # np_cv_rules = default_converter + numpy2ri.converter
#             eval_times_np = np.array(self.eval_times, dtype=np.float64)
    
#             for cid, cut in enumerate(cuts):
#                 tpr_at_time = []
#                 fpr_at_time = []
#                 ppv_at_time = []
#                 npv_at_time = []
#                 stats_at_time = []
    
#                 # SeSpPPVNPV was designed for constant risks. We have to adjust it for our time-
#                 # dependent risks: For each eval time, we use the corresponding risk column to compute
#                 # the metrics up to including the eval time and use the last value of the outputs.
#                 # numpy_cv = default_converter + numpy2ri.converter

#                 for tid, time in enumerate(eval_times_np):
#                     with conversion.localconverter(default_converter):
#                         numpy2ri.activate()
                        
#                         se_sp_ppv_npv = timeROC.SeSpPPVNPV(
#                             cutpoint=cut,
#                             T=self.times_np,
#                             delta=self.events_np,
#                             marker=risks[:, tid],
#                             cause=1,
#                             weighting="marginal",  # KM-IPCW
#                             times=eval_times_np[: tid + 1],
#                         )
#                         numpy2ri.deactivate()
    
#                         # Get the last values of the Se, Sp, PPV, NPV as these correspond to the
#                         # cumulated data up to the eval time for the risk column at the eval time.
#                         tpr_at_time.append(se_sp_ppv_npv.rx2("TP")[-1])
#                         fpr_at_time.append(se_sp_ppv_npv.rx2("FP")[-1])
#                         ppv_at_time.append(se_sp_ppv_npv.rx2("PPV")[-1])
#                         npv_at_time.append(se_sp_ppv_npv.rx2("NPV")[-1])
    
#                         stats_matrix = np.array(se_sp_ppv_npv.rx2("Stats"))
#                         row_names = list(se_sp_ppv_npv.rx2("Stats").rownames)
#                         col_names = list(se_sp_ppv_npv.rx2("Stats").colnames)
#                         stats_df = pd.DataFrame(
#                             stats_matrix, columns=col_names, index=row_names
#                         )
#                         stats_at_time.append(stats_df.iloc[-1])
    
#                         if cid == 0:
#                             # Get the naive ipcw weights from the statistics
#                             n_cumulative_cases_until_time = stats_at_time[-1]["Cases"]
#                             n_survivor_at_time = stats_at_time[-1]["survivor at t"]
#                             n_censored_at_time = stats_at_time[-1]["Censored at t"]
#                             n_controls_at_time = n_survivor_at_time + n_censored_at_time
#                             cum_n = n_controls_at_time + n_cumulative_cases_until_time
#                             weights.append(1.0 - (n_censored_at_time / cum_n))
    
#                             # Get the kaplan-meier estimate of the ipcw
#                             weights_km.append(
#                                 np.array(se_sp_ppv_npv.rx2("weights").rx2("IPCW.times"))[-1]
#                             )
    
#                 tprs.append(tpr_at_time)
#                 fprs.append(fpr_at_time)
#                 ppvs.append(ppv_at_time)
#                 npvs.append(npv_at_time)
#                 cutpoints.append(cut)
#                 stats.append(pd.DataFrame(stats_at_time))
    
#             # To (n_times, n_thresholds)
#             tprs = np.array(tprs).T
#             fprs = np.array(fprs).T
#             ppvs = np.array(ppvs).T
#             npvs = np.array(npvs).T
    
#             if threshold is not None:
#                 self.timeroc_tprs = tprs  # (n_times, )
#                 self.timeroc_fprs = fprs  # (n_times, )
#                 self.timeroc_ppvs = ppvs  # (n_times, )
#                 self.timeroc_npvs = npvs  # (n_times, )
#                 self.timeroc_naive_ipcw_weights = weights  # (n_times, )
#                 self.timeroc_ipcw_weights = weights_km  # (n_times, )
#             else:
#                 self.timeroc_tprs_1000 = tprs  # (n_times, n_thresholds)
#                 self.timeroc_fprs_1000 = fprs  # (n_times, n_thresholds)
#                 self.timeroc_ppvs_1000 = ppvs  # (n_times, n_thresholds)
#                 self.timeroc_npvs_1000 = npvs  # (n_times, n_thresholds)
#                 self.timeroc_stats = stats  # (n_thresholds, n_times)
#                 self.timeroc_cutpoints = cutpoints  # (n_thresholds, )
#                 self.timeroc_naive_ipcw_weights = weights  # (n_times, )
#                 self.timeroc_ipcw_weights = weights_km
    
#             return tprs, fprs, ppvs, npvs

    
#     def calc_rocs_timeroc(self):
#         if not hasattr(self, "timeroc_tprs_1000") or self.timeroc_tprs_1000 is None:
#             self.calc_SeSpPPVNPV_timeroc(threshold=None)  # (n_times, n_thresholds)
 
#         tprs_ = self.timeroc_tprs_1000
#         # print('passed tpts', tprs_)
#         fprs_ = self.timeroc_fprs_1000
#         # print('passed fprs_', fprs_)

 
#         n_times = np.array(tprs_).shape[0]
#         # print('n_times', n_times)
#         # assert n_times == len(
#         #     self.truncate_survtimes_preds()[0]
#         # ), "Number of times mismatch"
 
#         return tprs_, fprs_
 
#     def calc_precision_recall_curves_timeroc(self):
#         """Calculate precision and recall over n_thresholds for each eval_time. Uses the
#         model's survival predictions at the evaluation time! Is censoring adjusted 
#         using Kaplan-Meier
#         to predict the censoring distribution. Needs R installed and uses packages
#         timeROC and survival.
#         """
#         self.add_numpy_preds()
 
#         # survival_times, survs = self.truncate_survtimes_preds()
 
#         # Get tprs, fprs, ppvs, npvs
#         if hasattr(self, "timeroc_tprs_1000") and self.timeroc_tprs_1000 is not None:
#             tprs = self.timeroc_tprs_1000
#             fprs = self.timeroc_fprs_1000
#             ppvs = self.timeroc_ppvs_1000
#             npvs = self.timeroc_npvs_1000
#         else:
#             tprs, fprs, ppvs, npvs = self.calc_SeSpPPVNPV_timeroc(threshold=None)
 
#         recalls_ = np.array(tprs)
#         precisions_ = np.array(ppvs)
 
#         auprcs = []
#         precisions = []
#         recalls = []
#         for tid, time in enumerate(self.eval_times):
#             print(tid, time)
#             recall_ = recalls_[tid, :]
#             precision_ = precisions_[tid, :]
#             precision_[np.isnan(precision_)] = 0
 
#             # Ensure that curve values span the whole range of x axis for AUC calculation
#             recall_ = np.append(recall_, 0)
#             recall_ = np.insert(recall_, 0, values=1)
#             precision_ = np.insert(precision_, 0, values=0)
#             precision_ = np.append(precision_, 1)
 
#             precisions.append(precision_)
#             recalls.append(recall_)
 
#             auprc = auc(recall_, precision_)
#             auprcs.append(auprc)
 
#         self.pr_timeroc = (precisions, recalls, np.array(auprcs))
 
#         return precisions, recalls, np.array(auprcs)
 
#     def calc_auprcs(self, method="timeroc"):
#         """Get area under the precision-recall curve.
 
#         Args:
#             method (str): If "timeroc", uses Rs timeROC package and computes the AUPRC for each
#                 evaluation time using ipcw-adjusted sensitivity and precision estimates. It makes
#                 use of the model's survival predictions at the eval_time.
#                 If "sklearn", uses sklearn's precision_recall_curve function to compute the AUPRC,
#                 i.e. does not adjust for censoring. Also uses the model's survival predictions at
#                 the eval_time.
 
#         Returns:
#             auprcs (np.array): AUPRC at each evaluation time
#         """
#         if method == "timeroc":
#             if not "rpy2" in sys.modules:
#                 return np.nan
 
#             if hasattr(self, "pr_timeroc"):
#                 return self.pr_timeroc[2]
#             else:
#                 return self.calc_precision_recall_curves_timeroc()[2]
#         elif method == "sklearn":
#             return self.calc_precision_recall_curves_sklearn()[2]
#         else:
#             raise ValueError(
#                 f"Invalid method: {method}. Valid methods are: 'timeroc', 'sklearn'"
#             )
 
#     def calc_mean_auprc(self, method="timeroc"):
#         if method == "timeroc":
#             if not "rpy2" in sys.modules:
#                 return np.nan
 
#             if hasattr(self, "pr_timeroc"):
#                 return np.mean(self.pr_timeroc[2])
#             else:
#                 return np.mean(self.calc_precision_recall_curves_timeroc()[2])
#         elif method == "sklearn":
#             return np.mean(self.calc_precision_recall_curves_sklearn()[2])
#         else:
#             return np.nan
 
#     def get_aurocs_timeroc(self):
#             """The probability that a randomly chosen case (event by t) has a higher risk score than a
#             randomly chosen control (event after t). Uses the passed data to estimate ipc
#             (testing data). Uses cumulative/dynamic AUC."""
#             if not "rpy2" in sys.modules:
#                 return np.nan
    
#             tprs, fprs = self.calc_rocs_timeroc()
    
#             aucs = []
#             for tpr, fpr in zip(tprs, fprs):
#                 aucs.append(auc(fpr, tpr))
    
#             return np.array(aucs)
    
#     def get_auprcs_timeroc(self):
#             return self.calc_auprcs(method="timeroc")
