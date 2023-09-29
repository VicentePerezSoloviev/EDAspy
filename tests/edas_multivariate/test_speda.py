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
        speda = SPEDA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, lower_bound=-60, upper_bound=60,
                      alpha=0.5, l=10)

        assert speda.size_gen == 300
        assert speda.max_iter == 100
        assert speda.dead_iter == 20
        assert speda.n_variables == n_variables
        assert speda.alpha == 0.5
        assert speda.l_len == 10*int(speda.size_gen*speda.alpha)

    def test_list_bounds(self):
        l_bounds = [-(10 ** i) for i in range(10)]
        u_bounds = [10 ** i for i in range(10)]
        n_variables = len(l_bounds)
        eda = SPEDA(size_gen=300, max_iter=1, dead_iter=1, n_variables=n_variables, lower_bound=l_bounds,
                    upper_bound=u_bounds, alpha=0.5, disp=False, l=5)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        eda.minimize(benchmarking.cec14_4, False)
        data = eda.generation
        for i in range(n_variables):
            assert (l_bounds[i] <= data[:, i]).all() and (data[:, i] <= u_bounds[i]).all(), \
                "Lower bounds for dimension " + str(i) + " do not match."

    def test_archive(self):
        """
        Test if the archive is correct. When does not archive the maximum, and when it does and has to remove
        some solutions from the archive.
        """
        n_variables = 10
        speda = SPEDA(size_gen=300, max_iter=2, dead_iter=2, n_variables=10, lower_bound=-60, upper_bound=60,
                      alpha=0.5, l=10)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        speda.minimize(benchmarking.cec14_4, False)
        assert len(speda.archive) == int(speda.size_gen*speda.alpha)

        speda = SPEDA(size_gen=300, max_iter=15, dead_iter=15, n_variables=10, lower_bound=-60, upper_bound=60,
                      alpha=0.5, l=10)

        speda.minimize(benchmarking.cec14_4, False)
        assert len(speda.archive) == speda.l_len

    def test_check_generation(self):
        n_variables = 10
        speda = SPEDA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, lower_bound=-60, upper_bound=60,
                      alpha=0.5, l=10)

        gen = np.random.normal(
            [0] * speda.n_variables, [10] * speda.n_variables, [speda.size_gen, speda.n_variables]
        )
        speda.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        speda._check_generation(benchmarking.cec14_4)
        assert len(speda.evaluations) == len(speda.generation)

    def test_check_generation_parallel(self):
        n_variables = 10
        speda = SPEDA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, lower_bound=-60, upper_bound=60,
                      alpha=0.5, l=10, parallelize=True)

        gen = np.random.normal(
            [0] * speda.n_variables, [10] * speda.n_variables, [speda.size_gen, speda.n_variables]
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
        speda = SPEDA(size_gen=50, max_iter=100, dead_iter=20, n_variables=10, lower_bound=-60, upper_bound=60,
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

    def test_white_list(self):
        """
        Test if the white list is effective during runtime
        """
        n_variables = 10
        white_list = [('1', '2'), ('4', '5')]
        speda = SPEDA(size_gen=50, max_iter=10, dead_iter=10, n_variables=10, lower_bound=-60, upper_bound=60,
                      alpha=0.5, l=10, white_list=white_list)

        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        speda.minimize(benchmarking.cec14_4)
        assert all(elem in speda.pm.print_structure() for elem in white_list)

    def test_data_init(self):
        """
        Test if it is possible to initialize the EDA with custom data.
        """
        n_variables, size_gen, alpha = 10, 400, 0.5
        gen = np.random.normal(
            [0] * n_variables, [10] * n_variables, [size_gen, n_variables]
        )
        eda = SPEDA(size_gen=size_gen, max_iter=1, dead_iter=1, n_variables=n_variables, lower_bound=-60, upper_bound=60,
                    alpha=0.5, l=1, init_data=gen)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        eda.best_mae_global = 0  # to force breaking the loop when dead_iter = 1

        eda.minimize(benchmarking.cec14_4, output_runtime=False)

        assert (eda.generation == gen).all()

    def test_n_f_eval(self):
        """
        Test if the number of function evaluations in real
        """
        n_variables, size_gen, alpha, max_iter = 10, 100, 0.5, 10
        eda = SPEDA(size_gen=size_gen, max_iter=max_iter, dead_iter=10, n_variables=n_variables, alpha=alpha,
                    lower_bound=-60, upper_bound=60, l=10)
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

        eda = SPEDA(size_gen=size_gen, max_iter=max_iter, dead_iter=10, n_variables=n_variables, alpha=alpha,
                    landscape_bounds=(-60, 60), l=10)
        r = eda.minimize(problem, False)

        assert problem.state.evaluations == r.n_fev, "Number of function evaluations is not as expected"'''
