from unittest import TestCase
from EDAspy.optimization.custom import SPBN
import numpy as np
import pandas as pd


class TestSPBN(TestCase):

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

        gbn = SPBN(var_names)
        gbn.learn(data)
        samplings = gbn.sample(10)

        assert 1 < samplings[:, 0].mean() < 10
        assert 10 < samplings[:, 1].mean() < 100
        assert 100 < samplings[:, 2].mean() < 1000

    def test_sample_size(self):
        """
        Test if the samplings size is the expected.
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

        bn = SPBN(var_names)
        bn.learn(data)
        samplings = bn.sample(10)

        assert samplings.shape == (10, 3), "Shape does not match with the samplings."

    def test_white_list_spbn(self):
        np.random.seed(1234)
        n_vars = 30
        var_names = [str(i) for i in range(n_vars)]
        data = pd.DataFrame(np.random.random((100, n_vars)), columns=var_names)
        white_list = [('1', '2'), ('4', '5'), ('29', '15'), ('16', '7')]

        spbn = SPBN(var_names, white_list=white_list)
        spbn.learn(data)

        assert all(elem in spbn.print_structure() for elem in white_list)

    def test_black_list_spbn(self):
        np.random.seed(1234)
        n_vars = 15
        var_names = [str(i) for i in range(n_vars)]
        data = pd.DataFrame(np.random.random((1000, n_vars))*100, columns=var_names)
        data['pre1'] = np.random.random(1000)*100
        data['pre2'] = data['pre1'] * 2 + np.random.random(1000)
        black_list = [('pre1', 'pre2')]

        spbn = SPBN(var_names + ['pre1', 'pre2'], black_list=black_list)
        spbn.learn(data)

        assert not all(elem in spbn.print_structure() for elem in black_list)

    def test_black_and_white_list_spbn(self):
        np.random.seed(1234)
        n_vars = 30
        var_names = [str(i) for i in range(n_vars)]
        data = pd.DataFrame(np.random.random((1000, n_vars)) * 100, columns=var_names)
        data['pre1'] = np.random.random(1000) * 100
        data['pre2'] = data['pre1'] * 2 + np.random.random(1000)

        black_list = [('pre1', 'pre2')]
        white_list = [('1', '2'), ('4', '5'), ('29', '15'), ('16', '7')]

        spbn = SPBN(var_names + ['pre1', 'pre2'], black_list=black_list, white_list=white_list)
        spbn.learn(data)

        assert all(elem in spbn.print_structure() for elem in white_list)
        assert not all(elem in spbn.print_structure() for elem in black_list)