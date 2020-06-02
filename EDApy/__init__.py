#!/usr/bin/env python
# coding: utf-8

# __init__.py

import sys
if sys.version_info[0] < 3:
    raise Exception("Python version should be greater than 3")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.cluster import KMeans

from EDApy.optimization.multivariate.EDA_multivariate import EDAgbn as EDA_multivariate
from EDApy.optimization.univariate.discrete import UMDAd as EDA_discrete
from EDApy.optimization.univariate.continuous import UMDAc as EDA_continuous

from EDApy.optimization.multivariate.__BayesianNetwork import print_structure as print_structure
