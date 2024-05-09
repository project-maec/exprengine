import numpy as np
import pandas as pd


class Fitness:
    def __init__(self, function, greater_is_better: bool) -> None:
        self.function = function
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args) -> float:
        return self.function(*args)
    

def _sharpe_ratio(act_ret_label: pd.Series, pos: pd.Series, r_f: float | None = 0.0) -> float:
    # factors with no trading are considered the worst factors -> sharpe = np.nan
    close_copy = act_ret_label.copy()
    close_copy.index = pos.index
    benchmark_return = 0.
    daily_pl = pd.Series(np.nan_to_num(act_ret_label)*np.nan_to_num(pos)).groupby(act_ret_label.index.get_level_values(0)).sum()
    daily_total_pos = pd.Series(np.abs(np.nan_to_num(pos))).groupby(act_ret_label.index.get_level_values(0)).sum()
    ret = daily_pl/daily_total_pos.replace(0,1e20)
    
    if np.nansum(np.abs(ret)) <= 0:
        return np.nan
    return np.sqrt(250)*np.nanmean(ret)/np.nanstd(ret) 

def _mean_absolute_error(y: pd.Series, y_pred: pd.Series) -> float:
    return np.mean(np.abs(y_pred - y))

def _mean_square_error(y: pd.Series, y_pred: pd.Series) -> float:
    return np.mean(((y_pred - y) ** 2))

def _direction_accuracy(y: pd.Series, y_pred: pd.Series) -> float:
    close_direction = np.where(y > 0, 1, 0)
    factor_direction = np.where(y_pred > 0, 1, 0)
    return np.sum((close_direction == factor_direction)) / len(y_pred)


# ann_return = Fitness(_ann_return, greater_is_better=True)
sharpe_ratio = Fitness(_sharpe_ratio, greater_is_better=True)
mean_absolute_error = Fitness(_mean_absolute_error, greater_is_better=False)
mean_square_error = Fitness(_mean_square_error, greater_is_better=False)
direction_accuracy = Fitness(_direction_accuracy, greater_is_better=True)


fitness_map = {
    "sharpe ratio": sharpe_ratio,
    "mean absolute error": mean_absolute_error,
    "mean square error": mean_square_error,
    "direction accuracy": direction_accuracy,
}