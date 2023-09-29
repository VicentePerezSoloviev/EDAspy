from unittest import TestCase

from EDAspy.optimization.custom import UniGauss
import numpy as np


class TestUniGaussPM(TestCase):

    def test_print_structure_uni_gauss(self):
        n_vars = 30
        var_names = [str(i) for i in range(n_vars)]

        pm = UniGauss(variables=var_names, lower_bound=0.1)
        data = np.random.random((1000, n_vars))
        pm.learn(data)

        assert pm.print_structure() == list()

    def test_sample_size(self):
        """
        Test if the samplings size is the expected.
        """
        n_vars = 30
        var_names = [str(i) for i in range(n_vars)]

        pm = UniGauss(variables=var_names, lower_bound=0.1)
        data = np.random.random((1000, n_vars))
        pm.learn(data)
        samplings = pm.sample(10)

        assert samplings.shape == (10, n_vars), "Shape does not match with the samplings."

