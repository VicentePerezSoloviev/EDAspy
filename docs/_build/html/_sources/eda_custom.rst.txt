****************************************
Building my own EDA implementation
****************************************

In this notebook we show how the EDA can be implemented in a modular way using the components available in EDAspy.
This way, the user is able to build implementations that may not be considered in the state-of-the-art. EDASpy
also has the implementations of typical EDA implementations used in the state-of-the-art.

We first import from EDAspy all the needed functions and classes. To build our own EDA we use a modular class that
extends the abstract class of EDA used as a baseline of all the EDA implementations in EDAspy.

.. code-block:: python3

    from EDAspy.optimization.custom import EDACustom, GBN, UniformGenInit
    from EDAspy.benchmarks import ContinuousBenchmarkingCEC14

We initialize an object with the EDACustom object. Note that, independently of the pm and init parameteres,
we are goind to overwrite these with our own objects. If not, we have to choose which is the ID of the pm
and init we want.

.. code-block:: python3

    n_variables = 10
    my_eda = EDACustom(size_gen=100, max_iter=100, dead_iter=n_variables, n_variables=n_variables, alpha=0.5,
                       elite_factor=0.2, disp=True, pm=4, init=4, bounds=(-50, 50))

    benchmarking = ContinuousBenchmarkingCEC14(n_variables)

We now implement our initializator and probabilistic model and add these to our EDA.

.. code-block:: python3

    my_gbn = GBN([str(i) for i in range(n_variables)])
    my_init = UniformGenInit(n_variables)

    my_eda.pm = my_gbn
    my_eda.init = my_init

We run our EDA in one of the benchmarks that is implemented in EDAspy.

.. code-block:: python3

    eda_result = my_eda.minimize(cost_function=benchmarking.cec14_4)

We can access the results in the result object:

.. code-block:: python3

    print(eda_result)
