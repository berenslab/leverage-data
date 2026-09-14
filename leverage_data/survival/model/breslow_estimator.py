# Adapted from https://github.com/sebp/scikit-survival/blob/master/sksurv/linear_model/coxph.py
# to store the baseline hazard function and survival function and init the estimator from
# the saved data and clean data from negative durations.
#
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

import numpy as np

from .functions import StepFunction
from .non_parametric import _compute_counts
import torch


def fit_breslow(logits, t, e):
    """Maximum likelihood estimator for the cumulative baseline hazard function

    Args:
        model: model
        logits: log partial hazards (n, n_views, ) of your train or val set!
        t: time of events or censoring (n, 1, ) of your train or val set!
        e: binary event indicators (n, 1, ) of your train or val set!
    """

    logits = logits.squeeze()
    t = t.squeeze()
    e = e.squeeze()

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if isinstance(e, torch.Tensor):
        e = e.detach().cpu().numpy()

    # Clean durations: Keep only >= 0. Negative durations appear in datasets for multilabel
    # predictions. This breslow instance, however, only targets one label and we need to
    # remove the negative durations.
    idx = np.where(t < 0)[0]
    if len(idx) > 0:
        logits = np.delete(logits, idx)
        t = np.delete(t, idx)
        e = np.delete(e, idx)

    assert (
        np.unique(e).shape[0] == 2
    ), "e should be binary. Did you swap event and time?"

    return BreslowEstimator().fit(logits, e, t)


def init_breslow(cum_baseline_hazard, baseline_survival, unique_times):
    """Initialize the estimator from saved data.

    Parameters
    ----------
    cum_baseline_hazard : :class:`sksurv.functions.StepFunction`
        Cumulative baseline hazard function.

    baseline_survival : :class:`sksurv.functions.StepFunction`
        Baseline survival function.

    unique_times : ndarray
        Unique event times.

    Returns
    -------
    self
    """
    return BreslowEstimator().init(cum_baseline_hazard, baseline_survival, unique_times)


class BreslowEstimator:
    """Breslow's estimator of the cumulative hazard function. The StepFunction objects can be called
        to obtain a record's predicted values at the unique event times,
        e.g. get_cumulative_hazard_function(preds)[0]() returns the predicted cumulative hazard
        values for the first record at self.unique_times_.

    Attributes
    ----------
    cum_baseline_hazard_ : :class:`sksurv.functions.StepFunction`
        Cumulative baseline hazard function.

    baseline_survival_ : :class:`sksurv.functions.StepFunction`
        Baseline survival function.

    unique_times_ : ndarray
        Unique event times.
    """

    def init(self, cum_baseline_hazard, baseline_survival, unique_times):
        """Initialize the estimator from saved data.

        Parameters
        ----------
        cum_baseline_hazard : :class:`sksurv.functions.StepFunction`
            Cumulative baseline hazard function.

        baseline_survival : :class:`sksurv.functions.StepFunction`
            Baseline survival function.

        unique_times : ndarray
            Unique event times.

        Returns
        -------
        self
        """
        self.cum_baseline_hazard_ = cum_baseline_hazard
        self.baseline_survival_ = baseline_survival
        self.unique_times_ = unique_times
        return self

    def fit(self, linear_predictor, event, time):
        """Compute baseline cumulative hazard function.

        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.

        event : array-like, shape = (n_samples,)
            Contains binary event indicators.

        time : array-like, shape = (n_samples,)
            Contains event/censoring times.

        Returns
        -------
        self
        """
        risk_score = np.exp(linear_predictor)
        order = np.argsort(time, kind="mergesort")
        risk_score = risk_score[order]
        uniq_times, n_events, n_at_risk, _ = _compute_counts(event, time, order) #Count right censored and uncensored samples at each unique time point
        print(f"Unique event times from fit method: {uniq_times} n_events: {n_events} n_at_risk: {n_at_risk}")
        
        divisor = np.empty(n_at_risk.shape, dtype=float)
        value = np.sum(risk_score)
        divisor[0] = value
        k = 0
        for i in range(1, len(n_at_risk)):
            d = n_at_risk[i - 1] - n_at_risk[i]
            value -= risk_score[k : (k + d)].sum()
            k += d
            divisor[i] = value

        assert k == n_at_risk[0] - n_at_risk[-1]

        y = np.cumsum(n_events / divisor)
        self.cum_baseline_hazard_ = StepFunction(uniq_times, y)
        self.baseline_survival_ = StepFunction(uniq_times, np.exp(-y))
        self.unique_times_ = uniq_times

        return self

    def get_cumulative_hazard_function(self, linear_predictor):
        """Predict cumulative hazard function.

        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.

        Returns
        -------
        cum_hazard : ndarray, shape = (n_samples,)
            Predicted cumulative hazard functions with attributes x (times), y (predictions at
            times) and a (multiplicator for all entries)
            When calling a function, a is automatically applied to the y values.
        """
        risk_score = np.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(
                x=self.cum_baseline_hazard_.x,
                y=self.cum_baseline_hazard_.y,
                a=risk_score[i],
            )

        return funcs

    def get_survival_function(self, linear_predictor):
        """Predict survival function.

        Parameters
        ----------
        linear_predictor : array-like, shape = (n_samples,)
            Linear predictor of risk: `X @ coef`.

        Returns
        -------
        survival : ndarray, shape = (n_samples,)
            Predicted survival functions with attributes x (times) and y (predictions at times)
        """
        risk_score = np.exp(linear_predictor)
        n_samples = risk_score.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(
                x=self.baseline_survival_.x,
                y=np.power(self.baseline_survival_.y, risk_score[i]),
            )
        return funcs
