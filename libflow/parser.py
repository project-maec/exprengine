import re
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .function import Function
from .function import function_map
from .fitness import Fitness
from .syntaxtree import Node



class Parser:
    def __init__(self) -> None:
        """
        Vectorized factor backtesting (factor -> signal -> asset)
        [factor] outcome of SyntaxTree.execute(X)
        [signal] trading decision at the end of the datetime (>0: long, <0: short, =0: hold)
        [asset] backtesting result of an account applying the strategy
        """


    def parse_formula(expression):
        # This is a simplistic parser assuming well-formed input and does not handle errors
        if '(' in expression:
            name_expr = expression[:expression.find('(')]
            if name_expr in function_map.keys():
                # it's a function
                data = function_map[name_expr]
            elif name_expr.isnumeric():
                raise TypeError('numerical value is not callable')
            else:
                # it's data field name
                data = name_expr
            inside_parenthesis = expression[expression.find('(')+1:-1]
            args = []
            depth = 0
            last_split = 0
            # Split arguments considering nested functions
            for i, char in enumerate(inside_parenthesis):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                elif char == ',' and depth == 0:
                    args.append(parse_formula(inside_parenthesis[last_split:i].strip()))
                    last_split = i + 1
            args.append(parse_formula(inside_parenthesis[last_split:].strip()))
            return Node(data, args)
        else:
            print(expression)
            if expression.isnumeric():
                expression=float(expression)
            return Node(expression)