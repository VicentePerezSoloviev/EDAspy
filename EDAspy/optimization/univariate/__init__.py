#!/usr/bin/env python
# coding: utf-8

# __init__.py


from .discrete import UMDAd as EDA_discrete
from .continuous import UMDAc as EDA_continuous

from .umda_continuous import UMDAc
from .umda_binary import UMDAd

from ..eda import EDA
