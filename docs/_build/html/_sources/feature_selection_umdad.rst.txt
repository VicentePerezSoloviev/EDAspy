********************************************************
Using UMDAd for feature selection in a toy example
********************************************************

In this notebooks we show a toy example for feature selection using the binary implementation of EDA
in EDAspy. For this, we try to select the optimal subset of variables for a forecasting model. The
metric that we use for evaluation is the Mean Absolute Error (MAE) of the subset in the forecasting
model.

.. code-block:: python3

    # loading essential libraries first
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error

    # EDAspy libraries
    from EDAspy.optimization import UMDAd

We will use a small dataset to show an example of usage. We usually use a Feature Subset selector
when a great amount of variables is available to use.

.. code-block:: python3

    # import some data
    mdata = sm.datasets.macrodata.load_pandas().data
    df = mdata.iloc[:, 2:]
    df.head()

.. code-block:: python3

    variables = list(df.columns)
    variable_y = 'pop'  # pop is the variable we want to forecast
    variables = list(set(variables) - {variable_y})  # array of variables to select among transformations
    variables

We define a cost function which receives a dictionary with variables names as keys of the dictionary and
values 1/0 if they are used or not respectively.

The functions returns the Mean Absolute Error found with the combination of variables selected.


.. code-block:: python3

    def cost_function(variables_list, nobs=20, maxlags=10, forecastings=10):
    """
    variables_list: array of size the number of variables, where a 1 is to choose the variable, and 0 to
    reject it.
    nobs: how many observations for validation
    maxlags: previous lags used to predict
    forecasting: number of observations to predict

    return: MAE of the prediction with the real validation data
    """

    variables_chosen = []
    for i, j in zip(variables, variables_list):
        if j == 1:
            variables_chosen.append(i)

    data = df[variables_chosen + [variable_y]]

    df_train, df_test = data[0:-nobs], data[-nobs:]

    model = VAR(df_train)
    results = model.fit(maxlags=maxlags, ic='aic')

    lag_order = results.k_ar
    array = results.forecast(df_train.values[-lag_order:], forecastings)

    variables_ = list(data.columns)
    position = variables_.index(variable_y)

    validation = [array[i][position] for i in range(len(array))]
    mae = mean_absolute_error(validation, df_test['pop'][-forecastings:])

    return mae

We calculate the MAE found using all the variables.
This is an easy example so the difference between the MAE found using all the variables and the MAE
found after optimizing the model, will be very small. But this is appreciated with more difference
when large datasets are used.

.. code-block:: python3

    # build the dictionary with all 1s
    selection = [1]*len(variables)

    mae_pre_eda = cost_function(selection)
    print('MAE without using EDA:', mae_pre_eda)

We initialize the EDA weith the following parameters, and run the optimizer over the cost function
defined above. The vector of statistics is initialized to None so the EDA implementation will initialize
it. If you desire to initialize it in a way to favour some of the variables you can create a numpy array
with all the variables the same probability to be chosen or not (0.5), and the one you want to favour
to nearly 1. This will make the EDA to choose the variable nearly always.

.. code-block:: python3

    eda = UMDAd(size_gen=30, max_iter=100, dead_iter=10, n_variables=len(variables), alpha=0.5, vector=None,
            lower_bound=0.2, upper_bound=0.9, elite_factor=0.2, disp=True)

    eda_result = eda.minimize(cost_function=cost_function, output_runtime=True)

Note that the algorithm is minimzing correctly, but doe to the fact that it is a toy example, there is
not a high variance from the beginning to the end.

.. code-block:: python3

    print('Best cost found:', eda_result.best_cost)
    print('Variables chosen')
    variables_chosen = []
    for i, j in zip(variables, eda_result.best_ind):
            if j == 1:
                variables_chosen.append(i)
    print(variables_chosen)

We plot the best cost in each iteration to show how the MAE of the feature selection is reduced compared
to using all the variables.

.. code-block:: python3

    plt.figure(figsize = (14,6))

    plt.title('Best cost found in each iteration of EDA')
    plt.plot(list(range(len(eda_result.history))), eda_result.history, color='b')
    plt.xlabel('iteration')
    plt.ylabel('MAE')
    plt.show()
