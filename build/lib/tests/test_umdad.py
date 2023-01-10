from unittest import TestCase
from EDAspy.optimization import UMDAd
from EDAspy.benchmarks.binary import one_max
import numpy as np


class TestUMDAd(TestCase):

    def test_constructor(self):
        n_variables = 10
        umda = UMDAd(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_variables, alpha=0.5,
                     lower_bound=0.2, upper_bound=0.8, elite_factor=0.2)

        assert umda.size_gen == 100
        assert umda.max_iter == 100
        assert umda.dead_iter == 10
        assert umda.n_variables == n_variables
        assert umda.alpha == 0.5

    def test_bounds(self):
        """Check if new individuals meet the bounds restrictions"""
        n_variables = 10
        umda = UMDAd(size_gen=100, max_iter=2, dead_iter=0, n_variables=n_variables, alpha=0.5,
                     lower_bound=0.2, upper_bound=0.8, elite_factor=0.2, disp=False)

        umda.minimize(one_max, False)

        assert not np.any(umda.pm.pm < umda.lower_bound)
        assert not np.any(umda.pm.pm > umda.upper_bound)

    def test_vector(self):
        n_variables = 10
        umda = UMDAd(size_gen=100, max_iter=1, dead_iter=0, n_variables=n_variables, alpha=0.5,
                     lower_bound=-10, upper_bound=10, elite_factor=0.2)
        umda.minimize(one_max, False)

        assert len(umda.pm.pm) == n_variables

        vector = [0.5] * n_variables

        umda = UMDAd(size_gen=100, max_iter=1, dead_iter=0, n_variables=n_variables, alpha=0.5,
                     vector=vector, lower_bound=-10, upper_bound=10, elite_factor=0.2)
        umda.minimize(one_max, False)

        assert len(umda.vector) == n_variables

    def test_new_generation(self):
        n_variables = 10
        umda = UMDAd(size_gen=100, max_iter=1, dead_iter=0, n_variables=n_variables, alpha=0.5)

        umda.minimize(one_max, False)

        assert umda.generation.shape[0] == umda.size_gen + (umda.size_gen * umda.elite_factor)

    def test_check_generation(self):
        n_vars = 10
        vector = [0.5] * n_vars
        umda = UMDAd(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_vars, alpha=0.5, vector=vector)

        dataset = np.random.random((umda.size_gen, n_vars))
        dataset = dataset < vector
        dataset = np.array(dataset, dtype=int)

        umda.generation = dataset

        umda._check_generation(one_max)
        assert len(umda.evaluations) == len(umda.generation)

    def test_evaluate_solution(self):
        """
        Test if the generation is correctly evaluated, and the results are the same as if they are evaluated
        outside of the EDA framework.
        """
        n_variables = 10
        umda = UMDAd(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_variables, alpha=0.5)

        gen = np.random.normal(
            [0] * umda.n_variables, [10] * umda.n_variables, [umda.size_gen, umda.n_variables]
        )
        umda.generation = gen
        umda._check_generation(one_max)

        evaluations = []
        for sol in gen:
            evaluations.append(one_max(sol))

        assert (umda.evaluations == evaluations).all()

    def test_truncation(self):
        """
        Test if the size after truncation y correct
        """
        n_variables = 10
        umda = UMDAd(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_variables, alpha=0.5)

        gen = np.random.normal(
            [0] * umda.n_variables, [10] * umda.n_variables, [umda.size_gen, umda.n_variables]
        )
        umda.generation = gen
        umda._check_generation(one_max)

        umda._truncation()
        assert len(umda.generation) == int(umda.size_gen*umda.alpha)
