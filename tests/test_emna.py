from unittest import TestCase
from EDAspy.optimization import EMNA
from EDAspy.benchmarks import ContinuousBenchmarkingCEC14
import numpy as np


class TestEMNA(TestCase):

    def test_constructor(self):
        n_variables = 10
        emna = EMNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60), alpha=0.5)

        assert emna.size_gen == 300
        assert emna.max_iter == 100
        assert emna.dead_iter == 20
        assert emna.n_variables == n_variables
        assert emna.alpha == 0.5
        assert emna.landscape_bounds == (-60, 60)

    def test_new_generation(self):
        n_variables = 10
        emna = EMNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60))
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        emna.minimize(benchmarking.cec14_4, False)

        assert emna.generation.shape[0] == emna.size_gen + (emna.size_gen * emna.elite_factor)

    def test_check_generation(self):
        n_variables = 10
        emna = EMNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60))

        gen = np.random.normal(
            [0]*emna.n_variables, [10]*emna.n_variables, [emna.size_gen, emna.n_variables]
        )
        emna.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        emna._check_generation(benchmarking.cec14_4)
        assert len(emna.evaluations) == len(emna.generation)

    def test_evaluate_solution(self):
        """
        Test if the generation is correctly evaluated, and the results are the same as if they are evaluated
        outside of the EDA framework.
        """
        n_variables = 10
        emna = EMNA(size_gen=50, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60),
                    alpha=0.5)

        gen = np.random.normal(
            [0] * emna.n_variables, [10] * emna.n_variables, [emna.size_gen, emna.n_variables]
        )
        emna.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        emna._check_generation(benchmarking.cec14_4)

        evaluations = []
        for sol in gen:
            evaluations.append(benchmarking.cec14_4(sol))

        assert (emna.evaluations == evaluations).all()

    def test_truncation(self):
        """
        Test if the size after truncation y correct
        """
        n_variables = 10
        emna = EMNA(size_gen=50, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60),
                    alpha=0.5)

        gen = np.random.normal(
            [0] * emna.n_variables, [10] * emna.n_variables, [emna.size_gen, emna.n_variables]
        )
        emna.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        emna._check_generation(benchmarking.cec14_4)

        emna._truncation()
        assert len(emna.generation) == int(emna.size_gen*emna.alpha)

    def test_data_init(self):
        """
        Test if it is possible to initialize the EDA with custom data.
        """
        n_variables, size_gen, alpha = 10, 4, 0.5
        gen = np.random.normal(
            [0] * n_variables, [10] * n_variables, [size_gen, n_variables]
        )
        eda = EMNA(size_gen=size_gen, max_iter=1, dead_iter=1, n_variables=n_variables, landscape_bounds=(-60, 60),
                   alpha=0.5, init_data=gen)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        eda.best_mae_global = 0  # to force breaking the loop when dead_iter = 1

        evaluations = []
        for sol in gen:
            evaluations.append(benchmarking.cec14_4(sol))
        evaluations = np.array(evaluations)
        ordering = evaluations.argsort()
        best_indices_truc = ordering[: int(alpha * size_gen)]

        eda.minimize(benchmarking.cec14_4, output_runtime=False)

        assert (eda.generation == gen[best_indices_truc]).all()
