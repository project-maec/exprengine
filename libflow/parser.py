import re
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .function import Function
from .function import function_map
from .fitness import Fitness
from .syntax_tree import Node



class Parser:
    def __init__(self, list_dataname=[]) -> None:
        """
        @param list_dataname: list of data fields allowed
        """
        self.list_dataname = list_dataname


    def parse(self,expression):
        # This is a simplistic parser assuming well-formed input and does not handle errors
        if '(' in expression:
            name_expr = expression[:expression.find('(')]
            is_ts = False
            if name_expr in function_map.keys():
                # it's a function
                data = function_map[name_expr]
                if name_expr.startswith('ts_'):
                    is_ts =True
            elif name_expr.isnumeric():
                raise TypeError('numerical value is not callable')
            else:
                # it's data field name
                data = name_expr
            inside_parenthesis = expression[expression.find('(')+1:-1]
            args = []
            depth = 0
            last_split = 0
            
            # split arguments considering nested functions
            for i, char in enumerate(inside_parenthesis):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                elif char == ',' and depth == 0:
                    args.append(self.parse(inside_parenthesis[last_split:i].strip()))
                    last_split = i + 1
            args.append(self.parse(inside_parenthesis[last_split:].strip()))
            return Node(data, args, is_ts)
        else:
            if expression.isnumeric():
                expression=float(expression)
            else:
                if (not expression in self.list_dataname) and (len(self.list_dataname) > 0):
                    raise ValueError(f'{expression} not in allowed list of dataname')
            return Node(expression)