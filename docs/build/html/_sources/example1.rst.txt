
Hyper-parameter tunning
========================

A very easy function is to be optimized in this section to show how this EDA implementation works.

.. code-block:: python

    from EDAspy.optimization.univariate import EDA_continuous as EDAc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt


    # We must define a cost function to optimize. The cost function can be truncated.
    # For example, in the next defined, the values must be higher than 0. This can also be defined in the vector of statistics define below

    # define a cost function
    wheights = [20,10,-4]

    def cost_function(dictionary):
        """
        dictionary: python dictionary object with name of parameter as key, and value of the parameter as value
        return: total cost associated to the combination of parameters
        """
        function = wheights[0]*dictionary['param1']**2 + wheights[1]*(np.pi/dictionary['param2']) - 2 - wheights[2]*dictionary['param3']
        if function < 0:
            return 9999999
        return function


    # The vector of statistics define the starting values to start searching. We can define an initial mean and desviation. We can also define a maximum an minimum value for the hyper parameters. This can be defined in the cost function or in the vector of statistics.
    # If not desired to define the minimum and maximum, just delete the rows in pandas dataframe

    # initial vector of statistics
    vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
    vector['data'] = ['mu', 'std', 'min', 'max']
    vector = vector.set_index('data')
    vector.loc['mu'] = [5, 8, 1]
    vector.loc['std'] = 20

    vector.loc['min'] = 0  # optional
    vector.loc['max'] = 100  # optional

    # Execute the algorithm. We must define:
    # 1. Number of individuals in each generation
    # 2. Maximum number of iterations
    # 3. Number of iterations after which, if the cost is not improved, the algorithm finishes
    # 4. Percentage (over 1) of the population to be selected to mutate
    # 5. vector of statistics
    # 6. Aim: 'minimize' or 'maximize'
    # 7. The cost function to optimize
    #
    # The algorithm returns the best cost, a pandas dataframe with the solution found, and the history of costs

    EDA = EDAc(SIZE_GEN=40, MAX_ITER=200, DEAD_ITER=20, ALPHA=0.7, vector=vector,
                aim='minimize', cost_function=cost_function)
    bestcost, params, history = EDA.run()

    print('Best cost found:', bestcost)
    print('Best solution:\n', params)


    # # Some plots

    relative_plot = []
    mx = 999999999
    for i in range(len(history)):
        if history[i] < mx:
            mx = history[i]
            relative_plot.append(mx)
        else:
            relative_plot.append(mx)

    plt.figure(figsize = (14,6))

    ax = plt.subplot(121)
    ax.plot(list(range(len(history))), history)
    ax.title.set_text('Local cost found')
    ax.set_xlabel('iteration')
    ax.set_ylabel('MAE')

    ax = plt.subplot(122)
    ax.plot(list(range(len(relative_plot))), relative_plot)
    ax.title.set_text('Best global cost found')
    ax.set_xlabel('iteration')
    ax.set_ylabel('MAE')

    plt.show()

