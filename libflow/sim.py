import re
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import exchange_calendars as xcals
import glob



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
        # activate run_optim once we want to run a real backtest
        self.run_optim = False
        self.return_target = None
        self.delta_neutral = True
        self.signal_list = []
        self.startdate = 20150101
        self.enddate = 20230101
        self.calendar = ['US']
        self.is_period = ''
        self.expr_parser = Parser()

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

map_cal2mic={
    "US":["XNAS"],
    "JP":["XTKS"],
    "CN":["XSHG"]
}