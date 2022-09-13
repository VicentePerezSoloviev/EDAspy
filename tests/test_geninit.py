from unittest import TestCase
from EDAspy.optimization.custom.initialization_models import UniformGenInit, MultiGaussGenInit, \
    UniBinGenInit, UniGaussGenInit


class TestUMDAc(TestCase):
    n_variables = 10
    size_gen = 100

    def test_uniform(self):
        my_gen_init = UniformGenInit(self.n_variables)
        data = my_gen_init.sample(self.size_gen)

        assert data.shape == (self.size_gen, self.n_variables)

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
