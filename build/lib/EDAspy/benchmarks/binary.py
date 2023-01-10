#!/usr/bin/env python
# coding: utf-8

from typing import Union
import numpy as np


def one_max(array: Union[list, np.array]) -> Union[float, int]:
    """
    One max benchmark.
    :param array: solution to be evaluated in the cost function
    :return: evaluation of the solution
    """
    return sum(array)
