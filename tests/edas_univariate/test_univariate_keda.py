from unittest import TestCase
from EDAspy.optimization import UnivariateKEDA
from EDAspy.benchmarks import ContinuousBenchmarkingCEC14
import numpy as np


class TestUnivariateKEDA(TestCase):

    def test_constructor(self):
        n_variables = 10
        umda = UnivariateKEDA(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_variables, alpha=0.5,
                              elite_factor=0.2, lower_bound=-60, upper_bound=60)

        assert umda.size_gen == 100
        assert umda.max_iter == 100
        assert umda.dead_iter == 10
        assert umda.n_variables == n_variables
        assert umda.alpha == 0.5

    def test_list_bounds(self):
        l_bounds = [-(10**i) for i in range(10)]
        u_bounds = [10**i for i in range(10)]
        n_variables = len(l_bounds)
        eda = UnivariateKEDA(size_gen=300, max_iter=1, dead_iter=1, n_variables=n_variables, lower_bound=l_bounds,
                             upper_bound=u_bounds, alpha=0.5, disp=False)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        eda.minimize(benchmarking.cec14_4, False)
        data = eda.generation
        for i in range(n_variables):
            assert (l_bounds[i] <= data[:, i]).all() and (data[:, i] <= u_bounds[i]).all(), \
                "Lower bounds for dimension " + str(i) + " do not match."

    def test_new_generation(self):
        n_variables = 10
        umda = UnivariateKEDA(size_gen=100, max_iter=1, dead_iter=0, n_variables=n_variables, alpha=0.5,
                              lower_bound=-60, upper_bound=60)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        umda.minimize(benchmarking.cec14_4, False)

        assert umda.generation.shape[0] == umda.size_gen

    def test_check_generation(self):
        n_vars = 10
        umda = UnivariateKEDA(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_vars, alpha=0.5,
                              lower_bound=-60, upper_bound=60)

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
        umda = UnivariateKEDA(size_gen=100, max_iter=100, dead_iter=10, n_variables=n_variables, alpha=0.5,
                              lower_bound=-60, upper_bound=60)

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

    def test_data_init(self):
        """
        Test if it is possible to initialize the EDA with custom data.
        """
        n_variables, size_gen, alpha = 10, 40, 0.5
        gen = np.random.normal(
            [0] * n_variables, [10] * n_variables, [size_gen, n_variables]
        )
        eda = UnivariateKEDA(size_gen=size_gen, max_iter=1, dead_iter=1, n_variables=n_variables, alpha=alpha,
                             init_data=gen, lower_bound=-60, upper_bound=60)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        eda.best_mae_global = 0  # to force breaking the loop when dead_iter = 1

        eda.minimize(benchmarking.cec14_4, output_runtime=False)

        assert (eda.generation == gen).all()

    def test_n_f_eval(self):
        """
        Test if the number of function evaluations in real
        """
        n_variables, size_gen, alpha, max_iter = 10, 100, 0.5, 10
        eda = UnivariateKEDA(size_gen=size_gen, max_iter=max_iter, dead_iter=10, n_variables=n_variables, alpha=alpha,
                             lower_bound=-60, upper_bound=60)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        self.count = 0

        def f(sol):
            self.count += 1
            return benchmarking.cec14_4(sol)

        res = eda.minimize(f, output_runtime=False)
        print(self.count, res.n_fev, )

        assert self.count == res.n_fev, "Number of function evaluations is not as expected"

        '''import ioh
        problem = ioh.get_problem(
            "Sphere",
            instance=1,
            dimension=10,
            problem_class=ioh.ProblemClass.REAL
        )

        eda = UnivariateKEDA(size_gen=size_gen, max_iter=max_iter, dead_iter=10, n_variables=n_variables, alpha=alpha,
                             landscape_bounds=(-60, 60))
        r = eda.minimize(problem, False)

        assert problem.state.evaluations == r.n_fev, "Number of function evaluations is not as expected"'''