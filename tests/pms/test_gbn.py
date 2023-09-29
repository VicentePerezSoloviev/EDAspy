from unittest import TestCase
from EDAspy.optimization.custom import GBN
import numpy as np
import pandas as pd


class TestGBN(TestCase):

    np.random.seed(42)

    def test_sample(self):
        """
        Test if the samplings order is the same as the input used.
        """
        n_rows, n_vars = 10, 3
        var_names = ['A', 'B', 'C']

        data = pd.DataFrame(np.random.random((n_rows, n_vars)), columns=var_names)
        data['A'] *= 10
        data['A'] = abs(data['A'])
        data['B'] *= 100
        data['B'] = abs(data['B'])
        data['C'] *= 1000
        data['C'] = abs(data['C'])

        gbn = GBN(var_names)
        gbn.learn(data)
        samplings = gbn.sample(10)

        assert 1 < samplings[:, 0].mean() < 10
        assert 10 < samplings[:, 1].mean() < 100
        assert 100 < samplings[:, 2].mean() < 1000

    def test_white_list_gbn(self):
        np.random.seed(1234)
        n_vars = 30
        var_names = [str(i) for i in range(n_vars)]
        data = pd.DataFrame(np.random.random((100, n_vars)), columns=var_names)
        white_list = [('1', '2'), ('4', '5'), ('29', '15'), ('16', '7')]

        gbn = GBN(var_names, white_list=white_list)
        gbn.learn(data)

        assert all(elem in gbn.print_structure() for elem in white_list)

    def test_black_list_gbn(self):
        np.random.seed(1234)
        n_vars = 15
        var_names = [str(i) for i in range(n_vars)]
        data = pd.DataFrame(np.random.random((1000, n_vars)) * 100, columns=var_names)
        data['pre1'] = np.random.random(1000) * 100
        data['pre2'] = data['pre1'] * 2
        black_list = [('pre1', 'pre2')]

        gbn = GBN(var_names + ['pre1', 'pre2'], black_list=black_list)
        gbn.learn(data)

        assert not all(elem in gbn.print_structure() for elem in black_list)

    def test_black_and_white_list_gbn(self):
        np.random.seed(1234)
        n_vars = 30
        var_names = [str(i) for i in range(n_vars)]
        data = pd.DataFrame(np.random.random((1000, n_vars)) * 100, columns=var_names)
        data['pre1'] = np.random.random(1000) * 100
        data['pre2'] = data['pre1'] * 2

        black_list = [('pre1', 'pre2')]
        white_list = [('1', '2'), ('4', '5'), ('29', '15'), ('16', '7')]

        gbn = GBN(var_names + ['pre1', 'pre2'], black_list=black_list, white_list=white_list)
        gbn.learn(data)

        assert all(elem in gbn.print_structure() for elem in white_list)
        assert not all(elem in gbn.print_structure() for elem in black_list)
