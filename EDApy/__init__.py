#!/usr/bin/env python
# coding: utf-8

# __init__.py

import sys

if sys.version_info[0] < 3:
    raise Exception("Python version should be greater than 3")

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.cluster import KMeans

