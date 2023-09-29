from unittest import TestCase
from EDAspy.optimization import EBNA
import numpy as np


def categorical_cost_function(solution: np.array):
    """
    Example cost function that assigns costs to categorical choices.
    The cost function returns higher values for less desirable choices.
    """
    # Define a cost dictionary where each choice has an associated cost
    cost_dict = {
        'Color': {'Red': 0.1, 'Green': 0.5, 'Blue': 0.3},
        'Shape': {'Circle': 0.3, 'Square': 0.2, 'Triangle': 0.4},
        'Size': {'Small': 0.4, 'Medium': 0.2, 'Large': 0.1}
    }
    keys = list(cost_dict.keys())
    choices = {keys[i]: solution[i] for i in range(len(solution))}

    total_cost = 0.0
    for variable, choice in choices.items():
        total_cost += cost_dict[variable][choice]

    return total_cost


class TestEBNA(TestCase):

    variables = ['Color', 'Shape', 'Size']
    possible_values = np.array([
        ['Red', 'Green', 'Blue'],
        ['Circle', 'Square', 'Triangle'],
        ['Small', 'Medium', 'Large']], dtype=object
    )

    frequency = np.array([
        [.33, .33, .33],
        [.33, .33, .33],
        [.33, .33, .33]], dtype=object
    )

    def test_constructor(self):
        n_variables = 3
        eda = EBNA(size_gen=300, max_iter=100, dead_iter=20, n_variables=n_variables, alpha=0.5,
                   possible_values=self.possible_values, frequency=self.frequency)

        assert eda.size_gen == 300
        assert eda.max_iter == 100
        assert eda.dead_iter == 20
        assert eda.n_variables == n_variables
        assert eda.alpha == 0.5

    def test_new_generation(self):
        n_variables = 3
        eda = EBNA(size_gen=20, max_iter=10, dead_iter=10, n_variables=n_variables, alpha=0.5,
                   possible_values=self.possible_values, frequency=self.frequency)

        eda.minimize(categorical_cost_function, False)

        assert eda.generation.shape[0] == eda.size_gen

    def test_evaluate_solution(self):
        """
        Test if the generation is correctly evaluated, and the results are the same as if they are evaluated
        outside of the EDA framework.
        """
        n_variables = 3
        eda = EBNA(size_gen=20, max_iter=10, dead_iter=10, n_variables=n_variables, alpha=0.5,
                   possible_values=self.possible_values, frequency=self.frequency)

        gen = np.array([
            ['Red', 'Circle', 'Small'],
            ['Green', 'Square', 'Medium'],
            ['Blue', 'Triangle', 'Large']], dtype=object

        )
        eda.generation = gen
        eda._check_generation(categorical_cost_function)

        evaluations = []
        for sol in gen:
            evaluations.append(categorical_cost_function(sol))

        assert (eda.evaluations == evaluations).all()

    def test_data_init(self):
        """
        Test if it is possible to initialize the EDA with custom data.
        """
        gen = np.array([
            ['Red', 'Circle', 'Small'],
            ['Green', 'Square', 'Medium'],
            ['Blue', 'Triangle', 'Large']], dtype=object

        )
        n_variables = 3
        eda = EBNA(size_gen=20, max_iter=1, dead_iter=1, n_variables=n_variables, alpha=0.5,
                   possible_values=self.possible_values, frequency=self.frequency, init_data=gen)

        eda.best_mae_global = 999  # to force breaking the loop when dead_iter = 1

        eda.minimize(categorical_cost_function, output_runtime=False)

        assert (eda.generation == gen).all()

    def test_n_f_eval(self):
        """
        Test if the number of function evaluations in real
        """
        n_variables = 3
        eda = EBNA(size_gen=20, max_iter=1, dead_iter=1, n_variables=n_variables, alpha=0.5,
                   possible_values=self.possible_values, frequency=self.frequency)
        self.count = 0

        def f(sol):
            self.count += 1
            return categorical_cost_function(sol)

        res = eda.minimize(f, output_runtime=False)

        assert self.count == res.n_fev, "Number of function evaluations is not as expected"
