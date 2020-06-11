from unittest import TestCase


class TestUMDAd(TestCase):
    def test_check_generation(self):
        from EDAspy.optimization.univariate.discrete import UMDAd as EDAd
        import pandas as pd

        def check_solution_in_model(dictionary):
            cost = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
            return cost

        vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        vector.loc[0] = 0.5

        EDA = EDAd(MAX_IT=20, DEAD_ITER=10, SIZE_GEN=8, ALPHA=0.6, vector=vector, cost_function=check_solution_in_model,
                   aim='minimize')

        gen = pd.DataFrame(columns=vector.columns)

        individual = {'param1': 0, 'param2': 0, 'param3': 1}  # 1
        gen = gen.append(individual, True)
        individual = {'param1': 0, 'param2': 1, 'param3': 0}  # 2
        gen = gen.append(individual, True)
        individual = {'param1': 0, 'param2': 1, 'param3': 1}  # 3
        gen = gen.append(individual, True)
        individual = {'param1': 1, 'param2': 0, 'param3': 0}  # 4
        gen = gen.append(individual, True)
        individual = {'param1': 1, 'param2': 0, 'param3': 1}  # 5
        gen = gen.append(individual, True)
        individual = {'param1': 1, 'param2': 1, 'param3': 0}  # 6
        gen = gen.append(individual, True)
        individual = {'param1': 1, 'param2': 1, 'param3': 1}  # 7
        gen = gen.append(individual, True)
        individual = {'param1': 0, 'param2': 0, 'param3': 0}  # 8
        gen = gen.append(individual, True)

        expected_output = [1, 1, 2, 1, 2, 2, 3, 0]

        EDA.generation = gen
        EDA.check_generation()
        real_output = list(EDA.generation['cost'].values)

        assert real_output == expected_output, "Should be True"
        print("EDAspy.optimization.univariate.continuous.check_generation test passed")

    def test_individuals_selection(self):
        from EDAspy.optimization.univariate.discrete import UMDAd as EDAd
        import pandas as pd

        def check_solution_in_model(dictionary):
            cost = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
            return cost

        vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        vector.loc[0] = 0.5

        EDA = EDAd(MAX_IT=20, DEAD_ITER=10, SIZE_GEN=8, ALPHA=0.5, vector=vector, cost_function=check_solution_in_model,
                   aim='maximize')

        gen = pd.DataFrame(columns=vector.columns)

        individual = {'param1': 0, 'param2': 0, 'param3': 1, 'cost': 1}  # 1
        gen = gen.append(individual, True)
        individual = {'param1': 0, 'param2': 1, 'param3': 0, 'cost': 1}  # 2
        gen = gen.append(individual, True)
        individual = {'param1': 0, 'param2': 1, 'param3': 1, 'cost': 2}  # 3
        gen = gen.append(individual, True)
        individual = {'param1': 1, 'param2': 0, 'param3': 0, 'cost': 1}  # 4
        gen = gen.append(individual, True)
        individual = {'param1': 1, 'param2': 0, 'param3': 1, 'cost': 2}  # 5
        gen = gen.append(individual, True)
        individual = {'param1': 1, 'param2': 1, 'param3': 0, 'cost': 2}  # 6
        gen = gen.append(individual, True)
        individual = {'param1': 1, 'param2': 1, 'param3': 1, 'cost': 3}  # 7
        gen = gen.append(individual, True)
        individual = {'param1': 0, 'param2': 0, 'param3': 0, 'cost': 0}  # 8
        gen = gen.append(individual, True)

        expected_output = pd.DataFrame(columns=vector.columns)

        individual = {'param1': 1, 'param2': 1, 'param3': 1, 'cost': 3}  # 7
        expected_output = expected_output.append(individual, True)
        individual = {'param1': 0, 'param2': 1, 'param3': 1, 'cost': 2}  # 3
        expected_output = expected_output.append(individual, True)
        individual = {'param1': 1, 'param2': 0, 'param3': 1, 'cost': 2}  # 5
        expected_output = expected_output.append(individual, True)
        individual = {'param1': 1, 'param2': 1, 'param3': 0, 'cost': 2}  # 6
        expected_output = expected_output.append(individual, True)

        EDA.generation = gen
        EDA.individuals_selection()

        output = expected_output == EDA.generation
        for col in list(expected_output.columns):
            assert list(output[col]) == [True, True, True, True], "Should be True"
        print("EDAspy.optimization.univariate.continuous.individuals_selection test passed")

    def test_update_vector(self):
        from EDAspy.optimization.univariate.discrete import UMDAd as EDAd
        import pandas as pd

        def check_solution_in_model(dictionary):
            cost = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
            return cost

        vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        vector.loc[0] = 0.5

        EDA = EDAd(MAX_IT=20, DEAD_ITER=10, SIZE_GEN=8, ALPHA=0.5, vector=vector, cost_function=check_solution_in_model,
                   aim='maximize')

        gen = pd.DataFrame(columns=vector.columns)

        individual = {'param1': 1, 'param2': 1, 'param3': 1, 'cost': 3}  # 7
        gen = gen.append(individual, True)
        individual = {'param1': 0, 'param2': 1, 'param3': 1, 'cost': 2}  # 3
        gen = gen.append(individual, True)
        individual = {'param1': 1, 'param2': 0, 'param3': 1, 'cost': 2}  # 5
        gen = gen.append(individual, True)
        individual = {'param1': 1, 'param2': 1, 'param3': 0, 'cost': 2}  # 6
        gen = gen.append(individual, True)

        expected_output = pd.DataFrame(columns=vector.columns)
        expected_output.loc[0] = [3 / 4, 3 / 4, 3 / 4]

        EDA.generation = gen
        EDA.update_vector()
        output = expected_output == EDA.vector

        for col in list(vector.columns):
            assert list(output[col]) == [True], "Should be True"
        print("EDAspy.optimization.univariate.continuous.update_vector test passed")

    def test_check_individual(self):
        from EDAspy.optimization.univariate.discrete import UMDAd as EDAd
        import pandas as pd

        def check_solution_in_model(dictionary):
            cost = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
            return cost

        vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        vector.loc[0] = 0.5

        EDA = EDAd(MAX_IT=20, DEAD_ITER=10, SIZE_GEN=40, ALPHA=0.6, vector=vector,
                   cost_function=check_solution_in_model,
                   aim='minimize')

        individual = {'param1': 1, 'param2': 0, 'param3': 1}
        assert EDA.__check_individual__(individual) == 2, "Should be 2"
        print("EDAspy.optimization.univariate.continuous._check_individual_ test passed")

    def test_compare_costs(self):
        # assert sum((1, 2, 2)) == 6, "Should be 6"
        from EDAspy.optimization.univariate.discrete import UMDAd as EDAd
        import pandas as pd

        def check_solution_in_model(dictionary):
            cost = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
            return cost

        vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        vector.loc[0] = 0.5

        EDA = EDAd(MAX_IT=20, DEAD_ITER=10, SIZE_GEN=40, ALPHA=0.6, vector=vector,
                   cost_function=check_solution_in_model,
                   aim='minimize')

        EDA.best_MAE_global = 10
        assert EDA.__compare_costs__(8) is True, "Should be True"
        print("EDAspy.optimization.univariate.continuous.__compare_costs__ test passed")
