.. EDApy documentation master file, created by
   sphinx-quickstart on Fri Jun  5 11:10:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************************
EDApy
*********************************

.. toctree::
   :maxdepth: 2


Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Description of the package
##########################

EDApy is a Python package with different Estimation of Distribution algorithms implementations.

   1. Univariate binary EDA. Can be used for feature selection problems.

   2. Univariate continuous EDA. Parameter optimization.

   3. Multivariate continuous EDA. Complex problems in which dependencies among variables have to be modelled with a probabilistic model such as Gaussian Bayesian networks (in this case)

Easy examples
#############

Requirements
############

To use multivariate continuous EDA R installation is needed, with the following libraries c("bnlearn", "dbnR", "data.table").

To use both univariate EDAs there is no need to install R.

Documentation for the code
##########################

EDA multivariate with evidences
********************************
.. autoclass:: EDApy.optimization.multivariate.EDA_multivariate
.. autoclass:: EDApy.optimization.multivariate.EDA_multivariate.EDAgbn
   :members:

EDA multivariate with no evidences
***********************************
.. autoclass:: EDApy.optimization.multivariate.EDA_multivariate_gaussian
.. autoclass:: EDApy.optimization.multivariate.EDA_multivariate_gaussian.EDA_multivariate_gaussian
   :members:

EDA discrete
************
.. autoclass:: EDApy.optimization.univariate.EDA_discrete
.. autoclass:: EDApy.optimization.univariate.discrete.UMDAd
   :members:

EDA continuous
**************
.. autoclass:: EDApy.optimization.univariate.EDA_continuous
.. autoclass:: EDApy.optimization.univariate.continuous.UMDAc
   :members:
