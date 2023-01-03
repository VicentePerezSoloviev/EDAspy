from unittest import TestCase
from EDAspy.optimization import SPEDA
from EDAspy.benchmarks import ContinuousBenchmarkingCEC14
import numpy as np


class TestSPEDA(TestCase):

    def test_constructor(self):
        """
        Test the algorithm constructor, and if all the attributes are correctly set.
        """
        n_variables = 10
        speda = SPEDA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60),
                      alpha=0.5, l=10)

        assert speda.size_gen == 300
        assert speda.max_iter == 100
        assert speda.dead_iter == 20
        assert speda.n_variables == n_variables
        assert speda.alpha == 0.5
        assert speda.landscape_bounds == (-60, 60)
        assert speda.l_len == 10*int(speda.size_gen*speda.alpha)

    def test_archive(self):
        """
        Test if the archive is correct. When does not archive the maximum, and when it does and has to remove
        some solutions from the archive.
        """
        n_variables = 10
        speda = SPEDA(size_gen=300, max_iter=2, dead_iter=2, n_variables=10, landscape_bounds=(-60, 60),
                      alpha=0.5, l=10)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        speda.minimize(benchmarking.cec14_4, False)
        assert len(speda.archive) == 2*int(speda.size_gen*speda.alpha)

        speda = SPEDA(size_gen=300, max_iter=15, dead_iter=15, n_variables=10, landscape_bounds=(-60, 60),
                      alpha=0.5, l=10)

        speda.minimize(benchmarking.cec14_4, False)
        assert len(speda.archive) == speda.l_len

    def test_check_generation(self):
        n_variables = 10
        speda = SPEDA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60),
                      alpha=0.5, l=10)

        gen = np.random.normal(
            [0]*speda.n_variables, [10]*speda.n_variables, [speda.size_gen, speda.n_variables]
        )
        speda.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        speda._check_generation(benchmarking.cec14_4)
        assert len(speda.evaluations) == len(speda.generation)

    def test_evaluate_solution(self):
        """
        Test if the generation is correctly evaluated, and the results are the same as if they are evaluated
        outside of the EDA framework.
        """
        n_variables = 10
        speda = SPEDA(size_gen=50, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60),
                      alpha=0.5, l=10)

        gen = np.random.normal(
            [0] * speda.n_variables, [10] * speda.n_variables, [speda.size_gen, speda.n_variables]
        )
        speda.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        speda._check_generation(benchmarking.cec14_4)

        evaluations = []
        for sol in gen:
            evaluations.append(benchmarking.cec14_4(sol))

        assert (speda.evaluations == evaluations).all()

    def test_truncation(self):
        """
        Test if the size after truncation y correct
        """
        n_variables = 10
        speda = SPEDA(size_gen=50, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60),
                      alpha=0.5, l=10)

        gen = np.random.normal(
            [0] * speda.n_variables, [10] * speda.n_variables, [speda.size_gen, speda.n_variables]
        )
        speda.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        speda._check_generation(benchmarking.cec14_4)

        speda._truncation()
        assert len(speda.generation) == int(speda.size_gen*speda.alpha)

    def test_white_list(self):
        """
        Test if the white list is effective during runtime
        """
        n_variables = 10
        white_list = [('1', '2'), ('4', '5')]
        speda = SPEDA(size_gen=50, max_iter=10, dead_iter=10, n_variables=10, landscape_bounds=(-60, 60),
                      alpha=0.5, l=10, white_list=white_list)

        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        speda.minimize(benchmarking.cec14_4)
        assert all(elem in speda.pm.print_structure() for elem in white_list)
