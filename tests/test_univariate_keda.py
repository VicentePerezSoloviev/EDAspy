from unittest import TestCase
from EDAspy.optimization import UnivariateKEDA
from EDAspy.benchmarks import ContinuousBenchmarkingCEC14
import numpy as np


class TestUnivariateKEDA(TestCase):

    def test_constructor(self):
        n_variables = 10
        umda = UnivariateKEDA(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_variables, alpha=0.5,
                              elite_factor=0.2)

        assert umda.size_gen == 100
        assert umda.max_iter == 100
        assert umda.dead_iter == 10
        assert umda.n_variables == n_variables
        assert umda.alpha == 0.5

    def test_new_generation(self):
        n_variables = 10
        umda = UnivariateKEDA(size_gen=100, max_iter=1, dead_iter=0, n_variables=n_variables, alpha=0.5)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        umda.minimize(benchmarking.cec14_4, False)

        assert umda.generation.shape[0] == umda.size_gen + (umda.size_gen * umda.elite_factor)

    def test_check_generation(self):
        n_vars = 10
        umda = UnivariateKEDA(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_vars, alpha=0.5)

        gen = np.random.normal(
            [0]*umda.n_variables, [10]*umda.n_variables, [umda.size_gen, umda.n_variables]
        )
        umda.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_vars)
        umda._check_generation(benchmarking.cec14_4)
        assert len(umda.evaluations) == len(umda.generation)

    def test_evaluate_solution(self):
        """
        Test if the generation is correctly evaluated, and the results are the same as if they are evaluated
        outside of the EDA framework.
        """
        n_variables = 10
        umda = UnivariateKEDA(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_variables, alpha=0.5)

        gen = np.random.normal(
            [0] * umda.n_variables, [10] * umda.n_variables, [umda.size_gen, umda.n_variables]
        )
        umda.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        umda._check_generation(benchmarking.cec14_4)

        evaluations = []
        for sol in gen:
            evaluations.append(benchmarking.cec14_4(sol))

        assert (umda.evaluations == evaluations).all()

    def test_truncation(self):
        """
        Test if the size after truncation y correct
        """
        n_variables = 10
        umda = UnivariateKEDA(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_variables, alpha=0.5)

        gen = np.random.normal(
            [0] * umda.n_variables, [10] * umda.n_variables, [umda.size_gen, umda.n_variables]
        )
        umda.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        umda._check_generation(benchmarking.cec14_4)

        umda._truncation()
        assert len(umda.generation) == int(umda.size_gen*umda.alpha)

    def test_data_init(self):
        """
        Test if it is possible to initialize the EDA with custom data.
        """
        n_variables, size_gen, alpha = 10, 40, 0.5
        gen = np.random.normal(
            [0] * n_variables, [10] * n_variables, [size_gen, n_variables]
        )
        eda = UnivariateKEDA(size_gen=size_gen, max_iter=1, dead_iter=1, n_variables=n_variables, alpha=alpha,
                             init_data=gen)
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
