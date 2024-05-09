import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # prevent reporting 'All-NaN slice encountered'



class Function:
    def __init__(
        self, function, name: str, arity: int, func_type: int = 0, fixed_params: list = None
    ) -> None:
        self.function = function  # function
        self.name = name  # function name
        self.arity = arity  # number of function arguments
        # number of parameters forced to be constants
        self.func_type = func_type  # 0: basis function, 1:cross-sectional function, 2: time-series function
        # arguments forced to be certain variables
        self.fixed_params = [] if fixed_params is None else fixed_params

    def __call__(self, *args):
        return self.function(*args)