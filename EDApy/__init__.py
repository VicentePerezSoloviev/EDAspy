#!/usr/bin/env python
# coding: utf-8

# __init__.py

import sys

if sys.version_info[0] < 3:
    raise Exception("Python version should be greater than 3")

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.cluster import KMeans

from .optimization.multivariate.EDA_multivariate import EDAgbn as EDA_multivariate
from .optimization.univariate.discrete import UMDAd as EDA_discrete
from .optimization.univariate.continuous import UMDAc as EDA_continuous

from .optimization.multivariate.__BayesianNetwork import print_structure as print_structure
