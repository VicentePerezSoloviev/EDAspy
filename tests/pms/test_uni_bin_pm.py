from unittest import TestCase

from EDAspy.optimization.custom import UniBin
import numpy as np


class TestUniBinPM(TestCase):

    def test_print_structure_uni_bin(self):
        n_vars = 30
        var_names = [str(i) for i in range(n_vars)]

        pm = UniBin(variables=var_names, lower_bound=0.1, upper_bound=0.9)
        data = np.random.choice([0, 1], (1000, n_vars))
        pm.learn(data)

        assert pm.print_structure() == list()

    def test_bounds_uni_bin(self):
        n_vars = 30
        var_names = [str(i) for i in range(n_vars)]
        flag = False

        # case in which the upper bound is lower than the lower one
        try:
            UniBin(variables=var_names, lower_bound=0.9, upper_bound=0.1)
        except (Exception, ):
            flag = True

        assert flag

        # correct case
        UniBin(variables=var_names, lower_bound=0.1, upper_bound=0.9)

    def test_sample_size(self):
        """
        Test if the samplings size is the expected.
        """
        n_vars = 30
        var_names = [str(i) for i in range(n_vars)]

        pm = UniBin(variables=var_names, lower_bound=0.1, upper_bound=0.9)
        data = np.random.choice([0, 1], (1000, n_vars))
        pm.learn(data)
        samplings = pm.sample(10)

        assert samplings.shape == (10, n_vars), "Shape does not match with the samplings."
