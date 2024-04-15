from unittest import TestCase

from EDAspy.optimization.custom import UniCategorical, CategoricalSampling
import numpy as np


class TestUniCategoricalPM(TestCase):

    def test_print_structure_uni_gauss(self):
        variables = ['A', 'B', 'C']
        possible_values = np.array([
            ['q', 'w', 'e'],
            ['a', 's', 'd', 'f'],
            ['b', 'v']], dtype=object
        )

        frequency = np.array([
            [0.2, 0.5, 0.3],
            [.25, .25, .25, .25],
            [.1, .9]], dtype=object
        )

        n_rows = 100

        init = CategoricalSampling(n_variables=len(variables), possible_values=possible_values, frequency=frequency)
        data = init.sample(n_rows)

        pm = UniCategorical(variables=variables)
        pm.learn(data)

        assert pm.print_structure() == list()

    def test_sample_size(self):
        """
        Test if the samplings size is the expected.
        """
        variables = ['A', 'B', 'C']
        possible_values = np.array([
            ['q', 'w', 'e'],
            ['a', 's', 'd', 'f'],
            ['b', 'v']], dtype=object
        )

        frequency = np.array([
            [0.2, 0.5, 0.3],
            [.25, .25, .25, .25],
            [.1, .9]], dtype=object
        )

        n_rows = 100

        init = CategoricalSampling(n_variables=len(variables), possible_values=possible_values, frequency=frequency)
        data = init.sample(n_rows)

        pm = UniCategorical(variables=variables)
        pm.learn(data)
        samplings = pm.sample(10)

        assert samplings.shape == (10, len(variables)), "Shape does not match with the samplings."

