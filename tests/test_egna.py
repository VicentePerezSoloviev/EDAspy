from unittest import TestCase
from EDAspy.optimization import EGNA
from EDAspy.benchmarks import ContinuousBenchmarkingCEC14
import numpy as np


class TestUMDAc(TestCase):

    def test_constructor(self):
        n_variables = 10
        egna = EGNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60), alpha=0.5)

        assert egna.size_gen == 300
        assert egna.max_iter == 100
        assert egna.dead_iter == 20
        assert egna.n_variables == n_variables
        assert egna.alpha == 0.5
        assert egna.landscape_bounds == (-60, 60)

    def test_new_generation(self):
        n_variables = 10
        egna = EGNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60))
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        egna.minimize(benchmarking.cec14_4, False)

        assert egna.generation.shape[0] == egna.size_gen + (egna.size_gen * egna.elite_factor)

    def test_check_generation(self):
        n_variables = 10
        egna = EGNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60))

        gen = np.random.normal(
            [0]*egna.n_variables, [10]*egna.n_variables, [egna.size_gen, egna.n_variables]
        )
        egna.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        egna._check_generation(benchmarking.cec14_4)
        assert len(egna.evaluations) == len(egna.generation)

