********************************************************************************
Using EDAs for time series and times series transformation selection
********************************************************************************

When working with Time series in a Machine Learning project it is very common to try different
combinations of the time series in order to perform better the forecasting model. In this section
we will apply an EDA to select the optimal subset of variables and time series transformations to
improve the model.

.. code-block:: python3

    # loading essential libraries first
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error

    # EDAspy libraries
    from EDAspy.timeseries import EDA_ts_fts as EDA
    from EDAspy.timeseries import TSTransformations

.. code-block:: python3

    # import some data
    mdata = sm.datasets.macrodata.load_pandas().data
    df = mdata.iloc[:, 2:12]
    df.head()

.. code-block:: python3

    variables = list(df.columns)
    variable_y = 'pop'  # pop is the variable we want to forecast
    variables = list(set(variables) - {variable_y})  # array of variables to select among transformations
    variables

    We define a cost function which receives a dictionary with variables names as keys of the dictionary and
    values 1/0 if they are used or not respectively.

.. code-block:: python3

    TSTransf = TSTransformations(df)
    transformations = ['detrend', 'smooth', 'log']  # postfix to variables, to denote the transformation

    # build the transformations
    for var in variables:
        transformation = TSTransf.de_trending(var)
        df[var + 'detrend'] = transformation

    for var in variables:
        transformation = TSTransf.smoothing(var, window=10)
        df[var + 'smooth'] = transformation

    for var in variables:
        transformation = TSTransf.log(var)
        df[var + 'log'] = transformation

Define the cost function to calculate the Mean Absolute Error

.. code-block:: python3

    def cost_function(variables_list, nobs=20, maxlags=15, forecastings=10):
        """
        variables_list: list of variables without the variable_y
        nobs: how many observations for validation
        maxlags: previous lags used to predict
        forecasting: number of observations to predict

        return: MAE of the prediction with the real validation data
        """

        data = df[variables_list + [variable_y]]

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

We take the normal variables without any time series transformation and try to forecast
the y variable using the same cost function defined. This value is stored to be compared with
the optimum solution found

.. code-block:: python3

    eda = UMDAd(size_gen=30, max_iter=100, dead_iter=10, n_variables=len(variables), alpha=0.5, vector=None,
            lower_bound=0.2, upper_bound=0.9, elite_factor=0.2, disp=True)

    eda_result = eda.minimize(cost_function=cost_function, output_runtime=True)

Note that the algorithm is minimzing correctly, but doe to the fact that it is a toy example, there is
not a high variance from the beginning to the end.

.. code-block:: python3

    mae_pre_eda = cost_function(variables)
    print('MAE without using EDA:', mae_pre_eda)

Initialization of the initial vector of statitstics. Each variable has a 50% probability to be or not chosen

.. code-block:: python3

    vector = pd.DataFrame(columns=list(variables))
    vector.loc[0] = 0.5

Run the algorithm. The code will print some further information during execution

.. code-block:: python3

    eda = EDA(max_it=50, dead_it=5, size_gen=15, alpha=0.7, vector=vector,
          array_transformations=transformations, cost_function=cost_function)
    best_ind, best_MAE = eda.run(output=True)

We show some plots of the best solutions found during the execution in each iteration of the algorithm.

.. code-block:: python3

    # some plots
    hist = eda.historic_best

    relative_plot = []
    mx = 999999999
    for i in range(len(hist)):
        if hist[i] < mx:
            mx = hist[i]
            relative_plot.append(mx)
        else:
            relative_plot.append(mx)

    print('Solution:', best_ind, '\nMAE post EDA: %.2f' % best_MAE, '\nMAE pre EDA: %.2f' % mae_pre_eda)

    plt.figure(figsize = (14,6))

    ax = plt.subplot(121)
    ax.plot(list(range(len(hist))), hist)
    ax.title.set_text('Local cost found')
    ax.set_xlabel('iteration')
    ax.set_ylabel('MAE')

    ax = plt.subplot(122)
    ax.plot(list(range(len(relative_plot))), relative_plot)
    ax.title.set_text('Best global cost found')
    ax.set_xlabel('iteration')
    ax.set_ylabel('MAE')

    plt.show()
