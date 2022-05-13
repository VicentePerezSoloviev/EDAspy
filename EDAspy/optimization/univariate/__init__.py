#!/usr/bin/env python
# coding: utf-8

# __init__.py


import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_package(installed_pack, package):
    if package in installed_pack:
        return True
    else:
        return False


if sys.version_info[0] < 3:
    raise Exception("Python version should be greater than 3")

requirements = ['pandas', 'numpy', 'scipy']
reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

for package in requirements:
    if not check_package(installed_packages, package):
        print('package ', package, ' is not installed. Would you like to install it? y/n')
        response = input()
        if response == 'y' or response == 'yes':
            install(package)
        elif response == 'n' or response == 'no':
            print('dependencies not satisfied')


from .discrete import UMDAd as EDA_discrete
from .discrete import UMDAd as UMDA
from .continuous import UMDAc as EDA_continuous
from .continuous import UMDAc as UMDAc

