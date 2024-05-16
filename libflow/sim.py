import re
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import exchange_calendars as xcals
import glob
import datetime

from multiprocessing import Pool
from functools import partial

from .function import Function
from .function import function_map
from .fitness import Fitness
from .syntax_tree import Node
from .parser import Parser


class Simflow:
    def __init__(self, cfg: dict = {}) -> None:
        """
        @param cfg: dictionary for the sim parameters


        """
        self.univ = None
        # default values
        # activate run_optim once we want to run a real backtest
        self.run_optim = False
        self.return_target = None
        self.delta_neutral = True
        self.signal_dict = {}
        self.startdate = 20150101
        self.enddate = 20230101
        self.calendar = ['US']
        self.is_period = 'alt_q'
        self.expr_parser = Parser()
        self.verbose = 1

        for key, value in cfg.items():
            setattr(self, key, value)

        self.l_train_data = None
        self.l_signal_data = None
        self.train_label = None

    def set_target(self, Y):
        """
        @param Y: pd series for the required dates
        """

        return

    def add_alpha(self, alpha: str, alpha_name: str):
        alpha_expr = self.expr_parser.parse(alpha)
        self.signal_dict[alpha_name] = alpha_expr
        return

    def get_required_fields(self):
        list_fields = []
        for a in self.signal_dict.values():
            l_alpha_nodes = preorder_traversal(a)
            for e in l_alpha_nodes:
                if isinstance(e, str):
                    list_fields.append(e)
        return np.unique(list_fields).tolist()

    def get_in_sample_dates(self, dates):
        df_dates = pd.Series(dates)
        if self.is_period == 'alt_m':
            return df_dates[(df_dates.dt.quarter + df_dates.dt.year) % 2 != 0].tolist()
        elif self.is_period == 'alt_q':
            return df_dates[(df_dates.dt.month + df_dates.dt.year) % 2 != 0].tolist()
        elif '-' in self.is_period:
            return generate_date_list(self.is_period, df_dates)
        else:
            return dates

    def load_train_data(self, X_train):
        required_flds = self.get_required_fields()
        if len(X_train.index.names) > 1:
            X = X_train.reset_index()
        l_train_data = {}
        for x in required_flds:
            l_train_data[x] = X.pivot(index='date', columns='ticker', values=x)
        self.l_train_data = l_train_data

    def compute_all(self, X):
        l_signal_data = {}
        if self.l_signal_data is None:
            self.load_train_data(X)

        for sname in self.signal_dict.keys():
            if self.verbose > 0:
                print(f'computing {sname}')
            l_signal_data[sname] = self.signal_dict[sname](self.l_train_data)
        self.l_signal_data = l_signal_data
        return

    def score(self, sname, labels=None, sample_weights = None):
        signal_data = self.l_signal_data[sname]
        if not signal_data.index.equals(labels.index):
            raise ValueError(f'{sname}.index is different from Y.index')
        if not signal_data.columns.equals(labels.columns):
            raise ValueError(f'{sname}.columns is different from Y.columns')

        is_dates = self.get_in_sample_dates(signal_data.index)
        ic_series = row_wise_corr(signal_data.values,labels.values)
        ic_series.index = signal_data.index
        ic_series = ic_series[ic_series.index.isin(is_dates)]

        return get_information_table(ic_series)
    
    def score_os(self, sname, Y, sample_weights = None):
        signal_data = self.l_signal_data[sname]
        if not signal_data.index.equals(Y.index):
            raise ValueError(f'{sname}.index is different from Y.index')
        if not signal_data.columns.equals(Y.columns):
            raise ValueError(f'{sname}.columns is different from Y.columns')

        # is_dates = self.get_in_sample_dates(signal_data.index)
        ic_series = row_wise_corr(signal_data.values,Y.values)
        ic_series.index = signal_data.index
        # ic_series = ic_series[ic_series.index.isin(is_dates)]

        return get_information_table(ic_series)


map_cal2mic = {
    "US": ["XNAS"],
    "JP": ["XTKS"],
    "CN": ["XSHG"]
}


def get_holidays_by_calendar(calendar='US'):
    exch_list = map_cal2mic[calendar]
    df = pd.concat([get_holidays_by_exch(ex) for ex in exch_list])
    df = df['date'].drop_duplicates().tolist()
    return df


def get_holidays_by_exch(exch_mic):
    file_patts = f'/home/ubuntu/data/calendar/holidays/{exch_mic}_latest.csv'
    files = glob.glob(file_patts)
    if len(files) <= 0:
        return pd.DataFrame([], columns=['date', 'exchange'])
    else:
        return pd.read_csv(files[-1])


def preorder_traversal(node):
    list_elements = []
    if node is not None:
        list_elements.append(node.data)
        for child in node.children:
            list_elements.extend(preorder_traversal(child))
    return list_elements


def split_time_periods(periods_str):
    # Split the input string by semicolon to get individual periods
    periods = periods_str.split(';')

    # Initialize a list to hold the tuples of (start, end) dates
    time_periods = []

    # Iterate over each period and split by hyphen to get start and end dates
    for period in periods:
        start_date, end_date = period.split('-')
        time_periods.append((start_date, end_date))

    return time_periods


def generate_date_list(periods_str, dates: pd.Series):
    time_periods = split_time_periods(periods_str)
    all_dates = []

    for start_str, end_str in time_periods:
        # Convert string dates to datetime objects
        start_date = datetime.strptime(start_str, '%Y%m%d')
        end_date = datetime.strptime(end_str, '%Y%m%d')
        all_dates.append(dates[dates.between(start_date, end_date)].tolist())

    return all_dates


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


def get_information_table(ic_data):
    ic_data = ic_data.dropna()
    ic_summary_res = {}
    ic_summary_res["ic"] = ic_data.mean()
    ic_summary_res["ic_std"] = ic_data.std()
    ic_summary_res["ic_ir"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_res["t_stat"] = t_stat
    ic_summary_res["p_val"] = p_value
    ic_summary_res["ic_skew"] = stats.skew(ic_data)
    ic_summary_res["ic_kurt"] = stats.kurtosis(ic_data)

    return ic_summary_res

        
