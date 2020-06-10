.. EDAspy documentation master file, created by
   sphinx-quickstart on Fri Jun  5 11:10:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************************
EDAspy
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

EDAspy is a Python package with different Estimation of Distribution algorithms implementations.

   1. Univariate binary EDA. Can be used for feature selection problems.

   2. Univariate continuous EDA. Parameter optimization.

   3. Multivariate continuous EDA. Complex problems in which dependencies among variables have to be modelled with a probabilistic model such as Gaussian Bayesian networks (in this case)

   4. Multivariate continuous EDA with no evidences. New individuals are sampled from a multivariate gaussian distribution

Easy examples
#############

Requirements
############

To use multivariate continuous EDA using Bayesian networks, R installation is needed, with the following libraries c("bnlearn", "dbnR", "data.table").

To use both univariate EDAs there is no need to install R.

Documentation for the code
##########################

EDA multivariate with evidences
********************************
.. autoclass:: EDAspy.optimization.multivariate.EDA_multivariate
.. autoclass:: EDAspy.optimization.multivariate.EDA_multivariate.EDAgbn
   :members:

EDA multivariate with no evidences
***********************************
.. autoclass:: EDAspy.optimization.multivariate.EDA_multivariate_gaussian
.. autoclass:: EDAspy.optimization.multivariate.EDA_multivariate_gaussian.EDA_multivariate_gaussian
   :members:

EDA discrete
************
.. autoclass:: EDAspy.optimization.univariate.EDA_discrete
.. autoclass:: EDAspy.optimization.univariate.discrete.UMDAd
   :members:

EDA continuous
**************
.. autoclass:: EDAspy.optimization.univariate.EDA_continuous
.. autoclass:: EDAspy.optimization.univariate.continuous.UMDAc
   :members:
