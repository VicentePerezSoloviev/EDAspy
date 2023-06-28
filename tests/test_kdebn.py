from unittest import TestCase
from EDAspy.optimization.custom import KDEBN
import numpy as np
import pandas as pd


class TestKDEBN(TestCase):

    np.random.seed(42)

    def test_sample(self):
        """
        Test if the samplings order is the same as the input used.
        """
        n_rows, n_vars = 100, 3
        var_names = ['A', 'B', 'C']

        data = pd.DataFrame(np.random.random((n_rows, n_vars)), columns=var_names)
        data['A'] *= 10
        data['A'] = abs(data['A'])
        data['B'] *= 100
        data['B'] = abs(data['B'])
        data['C'] *= 1000
        data['C'] = abs(data['C'])

        gbn = KDEBN(var_names)
        gbn.learn(data)
        samplings = gbn.sample(10)

        assert 1 < samplings[:, 0].mean() < 10
        assert 10 < samplings[:, 1].mean() < 100
        assert 100 < samplings[:, 2].mean() < 1000
