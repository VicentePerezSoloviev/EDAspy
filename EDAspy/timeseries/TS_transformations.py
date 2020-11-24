import numpy as np
import matplotlib.pyplot as plt


class TSTransformations:
    data = -1

    # init function
    def __init__(self, data):
        self.data = data

    # remove trend from TS variable
    def de_trending(self, variable, plot=False):
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
        log_trans = np.log(self.data[variable])

        if plot:
            plt.plot(self.data[variable])
            plt.show()

            plt.plot(log_trans)
            plt.show()

        return list(log_trans)

    # Box-Cox transformation
    def box_cox(self, variable, lmbda, plot=False):
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
        rolling_mean = self.data[variable].rolling(window=window, min_periods=1).mean()

        if plot:
            plt.plot(rolling_mean)
            self.data[variable].plot()
            plt.show()

        return list(rolling_mean)

    # power transformations
    def power(self, variable, power, plot=False):
        values = self.data[variable].values
        transformation = [i**power for i in values]

        if plot:
            plt.plot(transformation)
            plt.plot(values)
            plt.show()

        return list(transformation)

    # exponential transformation
    def exponential(self, variable, numerator, plot=False):
        values = self.data[variable].values
        transformation = [np.exp(numerator/i) for i in values]

        if plot:
            plt.plot(transformation)
            plt.plot(values)
            plt.show()

        return transformation
