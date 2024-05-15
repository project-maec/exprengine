import re
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import exchange_calendars as xcals
import glob
import datetime


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
        self.signal_list = []
        self.startdate = 20150101
        self.enddate = 20230101
        self.calendar = ['US']
        self.is_period = 'alt_q'
        self.expr_parser = Parser()
        self.verbose = 1

        for key, value in cfg.items():
            setattr(self, key, value)

    def set_target(self, Y):
        """
        @param Y: pd series for the required dates
        """

        return

    def add_alpha(self, alpha: str):
        alpha_expr = self.expr_parser.parse(alpha)
        self.signal_list.append(alpha_expr)
        return

    def get_required_fields(self):
        list_fields = []
        for a in self.signal_list:
            l_alpha_nodes = preorder_traversal(a)
            for e in l_alpha_nodes:
                if isinstance(e,str):
                    list_fields.append(e)
        return np.unique(list_fields).tolist()
    

    def get_in_sample_dates(self, dates):
        df_dates = pd.Series(dates)
        if self.is_period == 'alt_m':
            return df_dates[(df_dates.dt.quarter + df_dates.dt.year)%2 != 0].tolist()
        elif self.is_period == 'alt_q':
            return df_dates[(df_dates.dt.month + df_dates.dt.year)%2 != 0].tolist()
        elif '-' in self.is_period:
            return generate_date_list(self.is_period, df_dates)
        else:
            return dates
        

    def compute(self,X):
        required_flds = self.get_required_fields()
        if len(X.index.names)>1:
            X = X.reset_index()
        l_train_data = {}
        for x in required_flds:
            l_train_data[x] = X.pivot(index='date',columns='ticker',values=x)
        self.l_train_data = l_train_data

        l_signal_data = {}
        for signal in self.signal_list:
            if self.verbose>0:
                print(f'computing {signal}')
            l_signal_data[signal] = signal(l_train_data)
        self.l_signal_data = l_signal_data
        return

    def score(self, X, Y):


        return

map_cal2mic={
    "US":["XNAS"],
    "JP":["XTKS"],
    "CN":["XSHG"]
}

def get_holidays_by_calendar(calendar='US'):
    exch_list = map_cal2mic[calendar]
    df = pd.concat([get_holidays_by_exch(ex) for ex in exch_list])
    df = df['date'].drop_duplicates().tolist()
    return df


def get_holidays_by_exch(exch_mic):
    file_patts = f'/home/ubuntu/data/calendar/holidays/{exch_mic}_latest.csv'
    files = glob.glob(file_patts)
    if len(files)<=0:
        return pd.DataFrame([], columns=['date','exchange'])
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


def generate_date_list(periods_str, dates:pd.Series):
    time_periods = split_time_periods(periods_str)
    all_dates = []
    
    for start_str, end_str in time_periods:
        # Convert string dates to datetime objects
        start_date = datetime.strptime(start_str, '%Y%m%d')
        end_date = datetime.strptime(end_str, '%Y%m%d')
        all_dates.append(dates[dates.between(start_date,end_date)].tolist())

    return all_dates