
import numpy as np
from survival_on_embedding.common.conversion_utils import e_t_to_tuple
from .metrics_auroc import cumulative_dynamic_auc
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score, brier_score

def get_ibs_sksurv(train_data, _events, _durations, survival_curves, evaluation_visits_from_now):
    ibs = integrated_brier_score(
        survival_train=e_t_to_tuple(train_data[0], train_data[1]),
        survival_test=e_t_to_tuple(_events, _durations),
        estimate=survival_curves, 
        times=evaluation_visits_from_now
        )
    return ibs

def get_brier_scores_sksurv(train_data, _events, _durations, survival_curves, evaluation_visits_from_now):
    bs = brier_score(
        e_t_to_tuple(train_data[0], train_data[1]),
        e_t_to_tuple(_events, _durations),
        survival_curves,
        evaluation_visits_from_now
    )[1]
    return bs

def get_aurocs_sksurv(train_data, _events, _durations, survival_curves, evaluation_visits_from_now):
    try:
        risks = 1.0 - np.array(survival_curves)
        aurocs = cumulative_dynamic_auc(
            e_t_to_tuple(train_data[0], train_data[1]),
            e_t_to_tuple(_events, _durations),
            risks,
            evaluation_visits_from_now,
        )[0]
    except ValueError as e:
        print(f"Error computing AUC: {e}")
        aurocs = np.nan
    return aurocs

def get_concordance_index_sksurv(_events, _durations, survival_curves, evaluation_visits_from_now):
    
    risks = risks = 1.0 - np.array(survival_curves) 
    sum_risk_c = (np.sum(risks, axis=1) if len(evaluation_visits_from_now) > 1 else risks.flatten())
    c_harrell = concordance_index_censored(np.array(_events).astype(bool), np.array(_durations), sum_risk_c)[0]

    return c_harrell

