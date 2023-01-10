*********
Changelog
*********

v1.1.0
======

- This version implements the SPEDA algorithm to allow dependencies between variables that fit Gaussian distributions and KDE nodes.
- This version implements a function to plot the BN structure learnt in the EDA implementations.
- This version enforces the tests to avoid bugs in the algorithms.
- This version implements the possibility of settings white and black boxes to set the mandatory or forbidden arcs in the BN structure learnt in each iteration.
- This version solves several bugs present in v1.0.2.

v1.0.2
======

- This version solves a bug in the EGNA optimizer related to the Gaussian Bayesian network structure learning.

v1.0.1
======

- This version solves a bug in the UMDAd optimizer related to the limits of the std in each variable.

v1.0.0
======

- This version implies a change in the way of using the EDAs.
- All EDAs extend an abstract class so, all EDAs have the some outline and the same minimize function.
- The cost function is now used only for the minimize function, so it is easier to be used.
- The probabilistic models and initialization models are treated separately from the EDA implementations so the user is able to decide whether to use a probabilistic model or other in the EDAs custom implementation.
- Th user is able to export and read the configuration of and EDA in order to re-use the same implementation in the future.
- All the EDA implementations have their own name according to the state-of-the-art of EDAs.
- More tests have been added.
- Documentation has been redone.
- Deprecation warning to TimeSeries selector. This class will be formatted. in following versions.
- The structure in the package has been removed and also the names.
- The implementation of EGAN with evidences has been removed to avoid having rpy2 as a dependency.

v0.2.0
======

- Time series transformations selection was added as a new functionality of the package.
- Added a notebooks section to show some real use cases of EDAspy. (3 implementations)

v0.1.2
======

- Added tests

v0.1.1
======

- Fixed bugs.
- Added documentation to readdocs.

v0.1.0
======

- First operative version 4 EDAs implemented.
- univariate EDA discrete.
- Univariate EDA continuous.
- Multivariate continuous EDA with evidences
- Multivariate continuous EDA with no evidences gaussian distribution.