import numpy as np
import matplotlib.pyplot as plt


class TSTransformations:
    """
    Tool to calculate time series transformations. Some time series transformations are given.
    This is just a very simple tool. It is not mandatory to use this tool to use the time series transformations
    selector. It is only disposed to be handy.
    """

    data = -1

    # init function
    def __init__(self, data):
        """
        Constructor of the class

        :param data: data to work with
        :type data: pandas DataFrame
        """
        self.data = data

    # remove trend from TS variable
    def de_trending(self, variable, plot=False):
        """
        Removes the trend of the time series.

        :param variable: string available in data DataFrame
        :type variable: string
        :param plot: if True plot is give, if False, not
        :type plot: bool
        :return: time series detrended
        :rtype: list
        """
        from sklearn.linear_model import LinearRegression

        total = []

        x = [i for i in range(0, len(self.data[variable]))]
        x = np.reshape(x, (len(x), 1))
        y = self.data[variable].values
        model = LinearRegression()
        model.fit(x, y)
        # calculate trend
        trend = model.predict(x)

        # de_trend
        de_trended = [y[i] - trend[i] for i in range(0, len(self.data[variable]))]
        [total.append(i) for i in de_trended]

        if plot:
            plt.plot(y)
            plt.plot(trend)
            plt.show()

            plt.plot(de_trended)
            plt.show()

        return total

    # calculate log of TS
    def log(self, variable, plot=False):
        """
        Calculate the logarithm of the time series.

        :param variable: name of variables
        :type variable: string
        :param plot: if True a plot is given.
        :type plot: bool
        :return: time series transformation
        :rtype: list
        """
        log_trans = np.log(self.data[variable])

        if plot:
            plt.plot(self.data[variable])
            plt.show()

            plt.plot(log_trans)
            plt.show()

        return list(log_trans)

    # Box-Cox transformation
    def box_cox(self, variable, lmbda, plot=False):
        """
        Calculate Box Cox time series transformation.

        :param variable: name of variable
        :type variable: string
        :param lmbda: lambda parameter of Box Cox transformation
        :type lmbda: float
        :param plot: if True, plot is given.ç
        :type plot: bool
        :return: time series transformation
        :rtype: list
        """
        from scipy.stats import boxcox
        transformed = boxcox(self.data[variable], lmbda=lmbda)

        if plot:
            plt.plot(self.data[variable])
            plt.show()

            plt.plot(transformed)
            plt.show()

        return list(transformed)

    # smooth of the TS, to reduce noise
    def smoothing(self, variable, window, plot=False):
        """
        Calculate time series smoothing transformation.

        :param variable: name of variable
        :type variable: string
        :param window: number of previous instances taken to smooth.
        :type window: int
        :param plot: if True, plot is given
        :type plot: bool
        :return: time series transformation
        :rtype: list
        """

        rolling_mean = self.data[variable].rolling(window=window, min_periods=1).mean()

        if plot:
            plt.plot(rolling_mean)
            self.data[variable].plot()
            plt.show()

        return list(rolling_mean)

    # power transformations
    def power(self, variable, power, plot=False):
        """
        Calculate power time series transformation.

        :param variable: name of variable
        :type variable: string
        :param power: exponent to calculate
        :type power: int
        :param plot: if True, plot is given
        :type plot: bool
        :return: time series transformation
        :rtype: list
        """

        values = self.data[variable].values
        transformation = [i**power for i in values]

        if plot:
            plt.plot(transformation)
            plt.plot(values)
            plt.show()

        return list(transformation)

    # exponential transformation
    def exponential(self, variable, numerator, plot=False):
        """
        Calculate exponential time series transformation.

        :param variable: name of variable
        :type variable: string
        :param numerator: numerator of the transformation
        :type numerator: int
        :param plot: if True, plot is given
        :type plot: bool
        :return: time series transformation
        :rtype: list
        """

        values = self.data[variable].values
        transformation = [np.exp(numerator/i) for i in values]

        if plot:
            plt.plot(transformation)
            plt.plot(values)
            plt.show()

        return transformation
