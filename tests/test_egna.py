from unittest import TestCase
from EDAspy.optimization import EGNA
from EDAspy.benchmarks import ContinuousBenchmarkingCEC14
import numpy as np


class TestEGNA(TestCase):

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
        egna = EGNA(size_gen=300, max_iter=1, dead_iter=1, n_variables=10, landscape_bounds=(-60, 60))
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

    def test_evaluate_solution(self):
        """
        Test if the generation is correctly evaluated, and the results are the same as if they are evaluated
        outside of the EDA framework.
        """
        n_variables = 10
        egna = EGNA(size_gen=50, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60),
                    alpha=0.5)

        gen = np.random.normal(
            [0] * egna.n_variables, [10] * egna.n_variables, [egna.size_gen, egna.n_variables]
        )
        egna.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        egna._check_generation(benchmarking.cec14_4)

        evaluations = []
        for sol in gen:
            evaluations.append(benchmarking.cec14_4(sol))

        assert (egna.evaluations == evaluations).all()

    def test_truncation(self):
        """
        Test if the size after truncation y correct
        """
        n_variables = 10
        egna = EGNA(size_gen=50, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60),
                    alpha=0.5)

        gen = np.random.normal(
            [0] * egna.n_variables, [10] * egna.n_variables, [egna.size_gen, egna.n_variables]
        )
        egna.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        egna._check_generation(benchmarking.cec14_4)

        egna._truncation()
        assert len(egna.generation) == int(egna.size_gen*egna.alpha)

    def test_white_list(self):
        """
        Test if the white list is effective during runtime
        """
        n_variables = 10
        white_list = [('1', '2'), ('4', '5')]
        egna = EGNA(size_gen=50, max_iter=10, dead_iter=10, n_variables=10, landscape_bounds=(-60, 60),
                    alpha=0.5, white_list=white_list)

        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        egna.minimize(benchmarking.cec14_4)
        assert all(elem in egna.pm.print_structure() for elem in white_list)
