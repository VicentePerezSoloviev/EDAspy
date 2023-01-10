from unittest import TestCase

import pandas as pd
from EDAspy.optimization.custom import GBN, SPBN
import numpy as np


class TestPM(TestCase):

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
