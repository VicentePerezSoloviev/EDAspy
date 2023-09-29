from unittest import TestCase
from EDAspy.optimization import MultivariateKEDA
from EDAspy.benchmarks import ContinuousBenchmarkingCEC14
import numpy as np


class TestMultivariateKEDA(TestCase):

    def test_constructor(self):
        """
        Test the algorithm constructor, and if all the attributes are correctly set.
        """
        n_variables = 10
        keda = MultivariateKEDA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, lower_bound=-60,
                                upper_bound=60, alpha=0.5, l=10)

        assert keda.size_gen == 300
        assert keda.max_iter == 100
        assert keda.dead_iter == 20
        assert keda.n_variables == n_variables
        assert keda.alpha == 0.5
        assert keda.l_len == 10*int(keda.size_gen*keda.alpha)

    def test_list_bounds(self):
        l_bounds = [-(10 ** i) for i in range(10)]
        u_bounds = [10 ** i for i in range(10)]
        n_variables = len(l_bounds)
        eda = MultivariateKEDA(size_gen=300, max_iter=1, dead_iter=1, n_variables=n_variables, lower_bound=l_bounds,
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
        keda = MultivariateKEDA(size_gen=300, max_iter=2, dead_iter=2, n_variables=10, lower_bound=-60, upper_bound=60,
                                alpha=0.5, l=10)
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        keda.minimize(benchmarking.cec14_4, False)
        assert len(keda.archive) == int(keda.size_gen*keda.alpha)

        keda = MultivariateKEDA(size_gen=300, max_iter=15, dead_iter=15, n_variables=10, lower_bound=-60, upper_bound=60,
                                alpha=0.5, l=5)

        keda.minimize(benchmarking.cec14_4, False)
        assert len(keda.archive) == keda.l_len

    def test_check_generation(self):
        n_variables = 10
        keda = MultivariateKEDA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, lower_bound=-60, upper_bound=60,
                                alpha=0.5, l=10)

        gen = np.random.normal(
            [0]*keda.n_variables, [10]*keda.n_variables, [keda.size_gen, keda.n_variables]
        )
        keda.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        keda._check_generation(benchmarking.cec14_4)
        assert len(keda.evaluations) == len(keda.generation)

    def test_evaluate_solution(self):
        """
        Test if the generation is correctly evaluated, and the results are the same as if they are evaluated
        outside of the EDA framework.
        """
        n_variables = 10
        keda = MultivariateKEDA(size_gen=50, max_iter=100, dead_iter=20, n_variables=10, lower_bound=-60, upper_bound=60,
                                alpha=0.5, l=10)

        gen = np.random.normal(
            [0] * keda.n_variables, [10] * keda.n_variables, [keda.size_gen, keda.n_variables]
        )
        keda.generation = gen
        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        keda._check_generation(benchmarking.cec14_4)

        evaluations = []
        for sol in gen:
            evaluations.append(benchmarking.cec14_4(sol))

        assert (keda.evaluations == evaluations).all()

    def test_white_list(self):
        """
        Test if the white list is effective during runtime
        """
        n_variables = 10
        white_list = [('1', '2'), ('4', '5')]
        keda = MultivariateKEDA(size_gen=50, max_iter=10, dead_iter=10, n_variables=10, lower_bound=-60, upper_bound=60,
                                alpha=0.5, l=10, white_list=white_list)

        benchmarking = ContinuousBenchmarkingCEC14(n_variables)

        keda.minimize(benchmarking.cec14_4)
        assert all(elem in keda.pm.print_structure() for elem in white_list)

    def test_kde_estimated_nodes(self):
        """
        Test if all the nodes learned during runtime have been estimated using KDE
        """
        n_variables = 10
        keda = MultivariateKEDA(size_gen=300, max_iter=2, dead_iter=0, n_variables=10, lower_bound=-60, upper_bound=60,
                                alpha=0.5, l=10)

        benchmarking = ContinuousBenchmarkingCEC14(n_variables)
        keda.minimize(benchmarking.cec14_4)

        # check if all variables have been estimated with CKDE
        for i in keda.pm.pm.nodes():
            assert str(keda.pm.pm.cpd(i).type()) == 'CKDEFactor'

    def test_n_f_eval(self):
        """
        Test if the number of function evaluations in real
        """
        n_variables, size_gen, alpha, max_iter = 10, 100, 0.5, 10
        eda = MultivariateKEDA(size_gen=size_gen, max_iter=max_iter, dead_iter=10, n_variables=n_variables, alpha=alpha,
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

        eda = MultivariateKEDA(size_gen=size_gen, max_iter=max_iter, dead_iter=10, n_variables=n_variables, alpha=alpha,
                   landscape_bounds=(-60, 60), l=10)
        r = eda.minimize(problem, False)

        assert problem.state.evaluations == r.n_fev, "Number of function evaluations is not as expected"'''
