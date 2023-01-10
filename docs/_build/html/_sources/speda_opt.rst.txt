****************************************
Using SPEDA for continuous optimization
****************************************

In this notebook we use the SPEDA approach to optimize a wellknown benchmark. Note that SPEDA learns and
sampled a semiparametric Bayesian network in each iteration. Import the algorithm and the benchmarks from
EDAspy.

.. code-block:: python3

    from EDAspy.optimization import SPEDA
    from EDAspy.benchmarks import ContinuousBenchmarkingCEC14

We will be using a benchmark with 10 variables.

.. code-block:: python3

    n_vars = 10
    benchmarking = ContinuousBenchmarkingCEC14(n_vars)

We initialize the EDA with the following parameters:

.. code-block:: python3

    speda = SPEDA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10,
                  landscape_bounds=(-60, 60), l=10)

    eda_result = speda.minimize(benchmarking.cec14_4, True)

We plot the best cost found in each iteration of the algorithm.

.. code-block:: python3

    plt.figure(figsize = (14,6))

    plt.title('Best cost found in each iteration of EDA')
    plt.plot(list(range(len(eda_result.history))), eda_result.history, color='b')
    plt.xlabel('iteration')
    plt.ylabel('MAE')
    plt.show()
