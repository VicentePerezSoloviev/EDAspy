****************************************
Using UMDAc for continuous optimization
****************************************

In this notebook we use the UMDAc implementation for the optimization of a cost function. This cost function
that we are using in this notebook is a wellknown benchmark and is available in EDAspy.

.. code-block:: python3

    from EDAspy.optimization.univariate import UMDAc
    from EDAspy.benchmarks import ContinuousBenchmarkingCEC14
    import matplotlib.pyplot as plt

We will be using 10 variables for the optimization.

.. code-block:: python3

    n_vars = 10
    benchmarking = ContinuousBenchmarkingCEC14(n_vars)

We initialize the EDA with the following parameters:

.. code-block:: python3

    umda = UMDAc(size_gen=100, max_iter=100, dead_iter=10, n_variables=10, alpha=0.5)
    # We leave bound by default
    eda_result = umda.minimize(cost_function=benchmarking.cec14_4, output_runtime=True)

We use the eda_result object to extract all the desired information from the execution.

.. code-block:: python3

    print('Best cost found:', eda_result.best_cost)
    print('Best solution:\n', eda_result.best_ind)

We plot the best cost in each iteration to show how the MAE of the feature selection is reduced compared
to using all the variables.

.. code-block:: python3

    plt.figure(figsize = (14,6))

    plt.title('Best cost found in each iteration of EDA')
    plt.plot(list(range(len(eda_result.history))), eda_result.history, color='b')
    plt.xlabel('iteration')
    plt.ylabel('MAE')
    plt.show()
