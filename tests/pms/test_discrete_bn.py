from unittest import TestCase
from EDAspy.optimization.custom import BN, CategoricalSampling
import numpy as np


class TestBN(TestCase):

    np.random.seed(42)

    def test_sample(self):
        """
        Test if the samplings order is the same as the input used.
        """
        variables = ['A', 'B', 'C']
        possible_values = np.array([
            ['q', 'w', 'e'],
            ['a', 's', 'd', 'f'],
            ['b', 'v']], dtype=object
        )

        frequency = np.array([
            [.25, .5, .25],
            [.25, .25, .25, .25],
            [.4, .6]], dtype=object
        )

        init = CategoricalSampling(n_variables=len(variables), possible_values=possible_values, frequency=frequency)
        data = init.sample(100)

        bn = BN(variables=variables)
        bn.learn(data)

        samples = bn.sample(100)

        for i in range(len(variables)):
            # check if unique values are the same
            assert len(set(possible_values[i]) - set(samples[:, i])) + \
                   len(set(samples[:, i]) - set(possible_values[i])) == 0, "Unique values and possible " \
                                                                           "values do not match"

    def test_sample_size(self):
        """
        Test if the samplings size is the expected.
        """
        variables = ['A', 'B', 'C']
        possible_values = np.array([
            ['q', 'w', 'e'],
            ['a', 's', 'd', 'f'],
            ['b', 'v']]
        )

        frequency = np.array([
            [0.2, 0.5, 0.3],
            [.25, .25, .25, .25],
            [.1, .9]]
        )

        n_rows = 100

        init = CategoricalSampling(n_variables=len(variables), possible_values=possible_values, frequency=frequency)
        data = init.sample(n_rows)

        bn = BN(variables=variables)
        bn.learn(data)

        samples = bn.sample(n_rows)

        assert samples.shape == (n_rows, len(variables)), "Shape does not match with the samplings."
