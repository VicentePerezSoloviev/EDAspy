from unittest import TestCase
from EDAspy.optimization.custom.initialization_models import UniformGenInit, MultiGaussGenInit, \
    UniBinGenInit, UniGaussGenInit, LatinHypercubeSampling, CategoricalSampling
import numpy as np


class TestUMDAc(TestCase):
    n_variables = 10
    size_gen = 100

    def test_uniform(self):
        my_gen_init = UniformGenInit(self.n_variables)
        data = my_gen_init.sample(self.size_gen)

        assert data.shape == (self.size_gen, self.n_variables)

    def test_uniform_bounds(self):
        l_bounds = [-(10**i) for i in range(self.n_variables)]
        u_bounds = [10**i for i in range(self.n_variables)]
        my_gen_init = UniformGenInit(self.n_variables, lower_bound=l_bounds, upper_bound=u_bounds)
        data = my_gen_init.sample(self.size_gen)

        for i in range(self.n_variables):
            assert (data[:, i] > l_bounds[i]).all() and (data[:, i] < u_bounds[i]).all(), "Bounds are not met."

    def test_multivariate_gaussian(self):
        my_gen_init = MultiGaussGenInit(self.n_variables)
        data = my_gen_init.sample(self.size_gen)

        assert data.shape == (self.size_gen, self.n_variables)

    def test_uni_bin(self):
        my_gen_init = UniBinGenInit(self.n_variables)
        data = my_gen_init.sample(self.size_gen)

        assert data.shape == (self.size_gen, self.n_variables)

    def test_uni_gauss(self):
        my_gen_init = UniGaussGenInit(self.n_variables)
        data = my_gen_init.sample(self.size_gen)

        assert data.shape == (self.size_gen, self.n_variables)

    def test_latin_hypercube_sampling(self):
        l_bounds = [-(10 ** i) for i in range(self.n_variables)]
        u_bounds = [10 ** i for i in range(self.n_variables)]
        my_gen_init = LatinHypercubeSampling(self.n_variables, lower_bound=l_bounds, upper_bound=u_bounds)
        data = my_gen_init.sample(self.size_gen)

        assert data.shape == (self.size_gen, self.n_variables)

    def test_latin_hypercube_sampling_bounds(self):
        l_bounds = [-(10**i) for i in range(self.n_variables)]
        u_bounds = [10**i for i in range(self.n_variables)]
        my_gen_init = LatinHypercubeSampling(self.n_variables, lower_bound=l_bounds, upper_bound=u_bounds)
        data = my_gen_init.sample(self.size_gen)

        for i in range(self.n_variables):
            assert (data[:, i] > l_bounds[i]).all() and (data[:, i] < u_bounds[i]).all(), "Bounds are not met."

    def test_categorical_sampling(self):
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

        my_gen_init = CategoricalSampling(len(variables), frequency=frequency, possible_values=possible_values)
        data = my_gen_init.sample(self.size_gen)

        assert data.shape == (self.size_gen, len(variables))
