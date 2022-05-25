from EDAspy.benchmarks import one_max
from EDAspy.optimization import UMDAc, UMDAd, EGNA
from EDAspy.benchmarks import ContinuousBenchmarkingCEC14
from EDAspy.optimization import PBILd, PBILc

def one_max_min(array):
    return -one_max(array)


n_vars = 10
benchmarking = ContinuousBenchmarkingCEC14(n_vars)


# umda = UMDAc(size_gen=100, max_iter=100, dead_iter=10, n_variables=10, alpha=0.5)
# umda = UMDAd(size_gen=100, max_iter=100, dead_iter=10, n_variables=10)
# a, b, c = umda.minimize(one_max_min, True)

# egna = EGNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=10, landscape_bounds=(-60, 60))
pbil = PBILc(size_gen=100, max_iter=300, dead_iter=100, n_variables=n_vars)

best_sol, best_cost, n_f_evals = pbil.minimize(cost_function=benchmarking.cec14_4)

print(best_sol)
print(best_cost)
print(n_f_evals)

