##### The following code is taken from sksurv/nonparametric.py and metrics.py.
##### We introduced one change: Adding a small epsilon to the denominator to avoid division by zero 
#####   in the function predict_ipcw. This mirrors the behavior of R's timeROC package and prevents
#####   rare errors in the computation of the IPCW.
EPSILON = 1e-7

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from sksurv.nonparametric import SurvivalFunctionEstimator, kaplan_meier_estimator, check_y_survival
from sksurv.metrics import _check_estimate_2d
import numpy as np


class CensoringDistributionEstimator(SurvivalFunctionEstimator):
    """Kaplan–Meier estimator for the censoring distribution."""

    def fit(self, y):
        """Estimate censoring distribution from training data.

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        event, time = check_y_survival(y)
        if event.all():
            self.unique_time_ = np.unique(time)
            self.prob_ = np.ones(self.unique_time_.shape[0])
        else:
            unique_time, prob = kaplan_meier_estimator(event, time, reverse=True)
            self.unique_time_ = np.r_[-np.inf, unique_time]
            self.prob_ = np.r_[1.0, prob]

        return self

    def predict_ipcw(self, y):
        """Return inverse probability of censoring weights at given time points.

        :math:`\\omega_i = \\delta_i / \\hat{G}(y_i)`

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        ipcw : array, shape = (n_samples,)
            Inverse probability of censoring weights.
        """
        event, time = check_y_survival(y)
        Ghat = self.predict_proba(time[event])

        Ghat = np.minimum(Ghat+EPSILON, 1.0)

        if (Ghat == 0.0).any():
            raise ValueError("censoring survival function is zero at one or more time points")

        weights = np.zeros(time.shape[0])
        weights[event] = 1.0 / Ghat

        return weights

def cumulative_dynamic_auc(survival_train, survival_test, estimate, times, tied_tol=1e-8):
    """Estimator of cumulative/dynamic AUC for right-censored time-to-event data.

    The receiver operating characteristic (ROC) curve and the area under the
    ROC curve (AUC) can be extended to survival data by defining
    sensitivity (true positive rate) and specificity (true negative rate)
    as time-dependent measures. *Cumulative cases* are all individuals that
    experienced an event prior to or at time :math:`t` (:math:`t_i \\leq t`),
    whereas *dynamic controls* are those with :math:`t_i > t`.
    The associated cumulative/dynamic AUC quantifies how well a model can
    distinguish subjects who fail by a given time (:math:`t_i \\leq t`) from
    subjects who fail after this time (:math:`t_i > t`).

    Given an estimator of the :math:`i`-th individual's risk score
    :math:`\\hat{f}(\\mathbf{x}_i)`, the cumulative/dynamic AUC at time
    :math:`t` is defined as

    .. math::

        \\widehat{\\mathrm{AUC}}(t) =
        \\frac{\\sum_{i=1}^n \\sum_{j=1}^n I(y_j > t) I(y_i \\leq t) \\omega_i
        I(\\hat{f}(\\mathbf{x}_j) \\leq \\hat{f}(\\mathbf{x}_i))}
        {(\\sum_{i=1}^n I(y_i > t)) (\\sum_{i=1}^n I(y_i \\leq t) \\omega_i)}

    where :math:`\\omega_i` are inverse probability of censoring weights (IPCW).

    To estimate IPCW, access to survival times from the training data is required
    to estimate the censoring distribution. Note that this requires that survival
    times `survival_test` lie within the range of survival times `survival_train`.
    This can be achieved by specifying `times` accordingly, e.g. by setting
    `times[-1]` slightly below the maximum expected follow-up time.
    IPCW are computed using the Kaplan-Meier estimator, which is
    restricted to situations where the random censoring assumption holds and
    censoring is independent of the features.

    This function can also be used to evaluate models with time-dependent predictions
    :math:`\\hat{f}(\\mathbf{x}_i, t)`, such as :class:`sksurv.ensemble.RandomSurvivalForest`
    (see :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Time-dependent-Risk-Scores>`).
    In this case, `estimate` must be a 2-d array where ``estimate[i, j]`` is the
    predicted risk score for the i-th instance at time point ``times[j]``.

    Finally, the function also provides a single summary measure that refers to the mean
    of the :math:`\\mathrm{AUC}(t)` over the time range :math:`(\\tau_1, \\tau_2)`.

    .. math::

        \\overline{\\mathrm{AUC}}(\\tau_1, \\tau_2) =
        \\frac{1}{\\hat{S}(\\tau_1) - \\hat{S}(\\tau_2)}
        \\int_{\\tau_1}^{\\tau_2} \\widehat{\\mathrm{AUC}}(t)\\,d \\hat{S}(t)

    where :math:`\\hat{S}(t)` is the Kaplan–Meier estimator of the survival function.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Area-under-the-ROC>`,
    [1]_, [2]_, [3]_ for further description.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples,) or (n_samples, n_times)
        Estimated risk of experiencing an event of test data.
        If `estimate` is a 1-d array, the same risk score across all time
        points is used. If `estimate` is a 2-d array, the risk scores in the
        j-th column are used to evaluate the j-th time point.

    times : array-like, shape = (n_times,)
        The time points for which the area under the
        time-dependent ROC curve is computed. Values must be
        within the range of follow-up times of the test data
        `survival_test`.

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    auc : array, shape = (n_times,)
        The cumulative/dynamic AUC estimates (evaluated at `times`).
    mean_auc : float
        Summary measure referring to the mean cumulative/dynamic AUC
        over the specified time range `(times[0], times[-1])`.

    See also
    --------
    as_cumulative_dynamic_auc_scorer
        Wrapper class that uses :func:`cumulative_dynamic_auc`
        in its ``score`` method instead of the default
        :func:`concordance_index_censored`.

    References
    ----------
    .. [1] H. Uno, T. Cai, L. Tian, and L. J. Wei,
           "Evaluating prediction rules for t-year survivors with censored regression models,"
           Journal of the American Statistical Association, vol. 102, pp. 527–537, 2007.
    .. [2] H. Hung and C. T. Chiang,
           "Estimation methods for time-dependent AUC models with survival data,"
           Canadian Journal of Statistics, vol. 38, no. 1, pp. 8–26, 2010.
    .. [3] J. Lambert and S. Chevret,
           "Summary measure of discrimination in survival models based on cumulative/dynamic time-dependent ROC curves,"
           Statistical Methods in Medical Research, 2014.
    """
    test_event, test_time = check_y_survival(survival_test)
    estimate, times = _check_estimate_2d(estimate, test_time, times, estimator="cumulative_dynamic_auc")

    n_samples = estimate.shape[0]
    n_times = times.shape[0]
    if estimate.ndim == 1:
        estimate = np.broadcast_to(estimate[:, np.newaxis], (n_samples, n_times))

    # fit and transform IPCW
    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw = cens.predict_ipcw(survival_test)

    # expand arrays to (n_samples, n_times) shape
    test_time = np.broadcast_to(test_time[:, np.newaxis], (n_samples, n_times))
    test_event = np.broadcast_to(test_event[:, np.newaxis], (n_samples, n_times))
    times_2d = np.broadcast_to(times, (n_samples, n_times))
    ipcw = np.broadcast_to(ipcw[:, np.newaxis], (n_samples, n_times))

    # sort each time point (columns) by risk score (descending)
    o = np.argsort(-estimate, axis=0)
    test_time = np.take_along_axis(test_time, o, axis=0)
    test_event = np.take_along_axis(test_event, o, axis=0)
    estimate = np.take_along_axis(estimate, o, axis=0)
    ipcw = np.take_along_axis(ipcw, o, axis=0)

    is_case = (test_time <= times_2d) & test_event
    is_control = test_time > times_2d
    n_controls = is_control.sum(axis=0)

    # prepend row of infinity values
    estimate_diff = np.concatenate((np.broadcast_to(np.inf, (1, n_times)), estimate))
    is_tied = np.absolute(np.diff(estimate_diff, axis=0)) <= tied_tol

    cumsum_tp = np.cumsum(is_case * ipcw, axis=0)
    cumsum_fp = np.cumsum(is_control, axis=0)
    true_pos = cumsum_tp / cumsum_tp[-1]
    false_pos = cumsum_fp / n_controls

    scores = np.empty(n_times, dtype=float)
    it = np.nditer((true_pos, false_pos, is_tied), order="F", flags=["external_loop"])
    with it:
        for i, (tp, fp, mask) in enumerate(it):
            idx = np.flatnonzero(mask) - 1
            # only keep the last estimate for tied risk scores
            tp_no_ties = np.delete(tp, idx)
            fp_no_ties = np.delete(fp, idx)
            # Add an extra threshold position
            # to make sure that the curve starts at (0, 0)
            tp_no_ties = np.r_[0, tp_no_ties]
            fp_no_ties = np.r_[0, fp_no_ties]
            scores[i] = np.trapz(tp_no_ties, fp_no_ties) #ify

    if n_times == 1:
        mean_auc = scores[0]
    else:
        surv = SurvivalFunctionEstimator()
        surv.fit(survival_test)
        s_times = surv.predict_proba(times)
        # compute integral of AUC over survival function
        d = -np.diff(np.r_[1.0, s_times])
        integral = (scores * d).sum()
        mean_auc = integral / (1.0 - s_times[-1])

    return scores, mean_auc
