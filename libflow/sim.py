import re
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .function import Function
from .function import function_map
from .fitness import Fitness
from .syntax_tree import Node
from .parser import Parser


class Simflow:
    def __init__(self, cfg:dict = {}) -> None:
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

        for key, value in cfg.items():
            setattr(self, key, value)


    def set_target(self, Y):
        """
        @param Y: pd series for the required dates
        """


        



    