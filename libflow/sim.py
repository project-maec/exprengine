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
        @param list_dataname: list of data fields allowed
        """
        self.univ = None
        # activate run_optim once we want to run a real backtest
        self.run_optim = False
        self.return_target = None
        self.delta_neutral = True


    