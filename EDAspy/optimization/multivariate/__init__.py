#!/usr/bin/env python
# coding: utf-8

# __init__.py

# __all__ = ['EDA_multivariate', '__BayesianNetwork']

import subprocess
import sys

import warnings
warnings.filterwarnings("ignore")


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_package(installed_pack, package):
    if package in installed_pack:
        return True
    else:
        return False


if sys.version_info[0] < 3:
    raise Exception("Python version should be greater than 3")

requirements = ['pandas', 'numpy', 'rpy2', 'pyvis', 'sklearn']
reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

for package in requirements:
    if not check_package(installed_packages, package):
        if package == 'sklearn':
            package = 'scikit-learn'

        print('package ', package, ' is not installed. Would you like to install it? y/n')
        response = input()
        if response == 'y' or response == 'yes':
            install(package)
        elif response == 'n' or response == 'no':
            print('dependencies insatisfied')


from .EDA_multivariate_gaussian import EDA_multivariate_gaussian as EDA_multivariate_gaussian
from .EDA_multivariate import EDAgbn as EDA_multivariate
from .__BayesianNetwork import print_structure as print_structure

