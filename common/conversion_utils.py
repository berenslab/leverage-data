import numpy as np
import pandas as pd
def e_t_to_tuple(
    e,
    t,
    time_type=float,
    event_col_name="event",
    time_col_name="time",
    order=["event", "time"],
):
    """Convert event and time to structured array.

    Args:
        e (np.array): Array of events
        t (np.array): Array of times
        time_type (type): Type of time. Can be int or float
        event_col_name (str): Name of event column. Defaults to "event"
        time_col_name (str): Name of time column. Defaults to "time"
        order (list): Wanted order of event and time in tuple. Defaults to ["event", "time"]

    Returns:
        np.array: Structured array of events and times in the specified order
    """

    assert len(e) == len(t), "e and t must have the same length"
    assert order == ["event", "time"] or order == [
        "time",
        "event",
    ], "order must be ['event', 'time'] or ['time', 'event']"
    if order == ["time", "event"]:
        t_ = t.copy()
        t = e.copy()
        e = t_.copy()
    if time_type == int:
        return np.array(
            [(e[i], int(t[i])) for i in range(len(e))],
            dtype=[(event_col_name, bool), (time_col_name, int)],
        )
    return np.array(
        [(e[i], t[i]) for i in range(len(e))],
        dtype=[(event_col_name, bool), (time_col_name, float)],
    )

def et_tuple_to_df(et_tuple, event_col_name="event", time_col_name="time"):
    """Convert structured array to dataframe.
    Uses column names "event" and "time" by default.
    """

    return pd.DataFrame(et_tuple, columns=[event_col_name, time_col_name])
