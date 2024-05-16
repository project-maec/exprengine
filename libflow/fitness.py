import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool
from functools import partial

class Fitness:
    def __init__(self, function, greater_is_better: bool) -> None:
        self.function = function
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args) -> float:
        return self.function(*args)
    

### Helper functions

def compute_spearman_corr(row_index, arr1, arr2):
    return [row_index, stats.spearmanr(arr1[row_index], arr2[row_index], nan_policy='omit')[0]]

def row_wise_corr(arr1, arr2, num_workers=4):
    # row_correlations = np.zeros(arr1.shape[0])
    with Pool(num_workers) as pool:
        res = pool.map(partial(compute_spearman_corr, arr1=arr1,
                       arr2=arr2), range(arr1.shape[0]))
        pool.close()
    df_res = pd.DataFrame(
        res, columns=['idx', 'val']).sort_values('idx')['val']
    return pd.Series(df_res.values,name='daily_ic')

### metrics

def _sharpe_ratio_legacy(act_ret_label: pd.Series, pos: pd.Series, r_f: float | None = 0.0) -> float:
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

def _sharpe_ratio(y,y_pred) -> float:
    y_copy = y.copy()
    if not y_pred.index.equals(y.index):
        raise ValueError('y_pred.index is different from y.index')
    if not y_pred.columns.equals(y.columns):
        raise ValueError('y_pred.columns is different from Y.columns')
    daily_pl = np.nan_to_num(y)*np.nan_to_num(y_pred)
    daily_pl = (np.nan_to_num(y)*np.nan_to_num(y_pred)).sum(axis=1)
    daily_total_pos = np.abs(np.nan_to_num(y_pred)).sum(axis=1)
    ret = daily_pl/np.where(daily_total_pos<=0,1e20,daily_total_pos)

    if np.nansum(np.abs(ret)) <= 0:
        return np.nan
    return np.sqrt(250)*np.nanmean(ret)/np.nanstd(ret)


def _ic(y, y_pred) -> float:
    if not y_pred.index.equals(y.index):
        raise ValueError('y_pred.index is different from y.index')
    if not y_pred.columns.equals(y.columns):
        raise ValueError('y_pred.columns is different from Y.columns')
    ic_series = row_wise_corr(y.values,y_pred.values)
    
    return np.nanmean(ic_series)

def _icir(y,y_pred)-> float:
    if not y_pred.index.equals(y.index):
        raise ValueError('y_pred.index is different from y.index')
    if not y_pred.columns.equals(y.columns):
        raise ValueError('y_pred.columns is different from Y.columns')
    ic_series = row_wise_corr(y.values,y_pred.values)

    if np.nanstd(ic_series)<=0:
        return np.nan
    return np.nanmean(ic_series)/np.nanstd(ic_series)

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
icir = Fitness(_icir, greater_is_better=True)
ic = Fitness(_ic, greater_is_better=True)
mean_absolute_error = Fitness(_mean_absolute_error, greater_is_better=False)
mean_square_error = Fitness(_mean_square_error, greater_is_better=False)
direction_accuracy = Fitness(_direction_accuracy, greater_is_better=True)


fitness_map = {
    "sr": sharpe_ratio,
    "icir": icir,
    "ic":ic,
    "mae": mean_absolute_error,
    "mse": mean_square_error,
    "direction_accuracy": direction_accuracy,
}