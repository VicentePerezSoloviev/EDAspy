Introduction
============

Estimation of Distribution Algorithms (EDA) are a type of evolutionary algorithms that incorporate a probabilistic model to reproduce the next generations of the algorithm.
Depending on the type of EDA, the relations among variables is different, and different probabilistic models are used.

EDAspy is a Python package with different EDA implementations. This is the first and unique package created and full dedicated to this type of algorithms.

Motivation
**********

The aim of this package is to continue adding different implementations of this algorithm. Also, different implementations specifically designed to use
in some Machine Learning projects (time series for example) are implemented.

Feel free to contribute in this package with more implementations of the algorithm.

Actual implementations
***********************

The following EDAs have been implemented:

General EDAs:

    1. Binary univariate EDA. It can be used as a simple example of EDA, or to use it for feature selection.

    2. Continuous univariate EDA. It can be used for hyper-parameter optimization or continuous function optimization.

    3. Continuous multivariate EDA. Using Gaussian Bayesian Networks to model an abstract representation of the search space. Can be used to continuous multivariate optimization problems.

    4. Continuous multivariate EDA. Another approach not using GBNs that can be used for hyper-parameter optimization, or other optimization problems.

Specific EDAs:

    1. Binary multivariate EDA. This approach selects the best time series transformation to improve the model forecasting performance.

Installation
*************

EDAspy can be downloaded from PyPi using pip command:

.. code-block:: bash

    pip install EDAspy

Depending on the algorithms you want to run, different requisites are needed to be installed.
To use the optimization.multivariate approach using Bayesian Networks, an interface to connect with R is needed. EDaspy
uses rpy2 package to do it. EDAspy uses bnlearn package implemented in R to manage different functions with Bayesian Networks.

To install rpy2:

.. code-block:: bash

    pip install rpy2

To install R packages:

.. code-block:: R

    install.packages(c("bnlearn", "data.table", "dbnR"))