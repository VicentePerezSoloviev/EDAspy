#!/usr/bin/env python
# coding: utf-8


def test__compare_costs__():
    # assert sum((1, 2, 2)) == 6, "Should be 6"
    from EDAspy.optimization.univariate.continuous import UMDAc as EDAc
    import pandas as pd

    def cost_function(dictionary):
        function = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
        if function < 0:
            return 9999999
        return function

    vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
    vector['data'] = ['mu', 'std', 'min', 'max']
    vector = vector.set_index('data')
    vector.loc['mu'] = [5, 8, 1]
    vector.loc['std'] = 20
    vector.loc['min'] = 0
    vector.loc['max'] = 100

    EDA = EDAc(MAX_ITER=20, DEAD_ITER=10, SIZE_GEN=40, ALPHA=0.6, vector=vector, cost_function=cost_function,
               aim='minimize')

    EDA.best_MAE_global = 10.0
    assert EDA.__compare_costs__(8.0) is True, "Should be True"


def test_check_individual_():
    # assert sum((1, 2, 2)) == 6, "Should be 6"
    from EDAspy.optimization.univariate.continuous import UMDAc as EDAc
    import pandas as pd

    def cost_function(dictionary):
        function = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
        if function < 0:
            return 9999999
        return function

    vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
    vector['data'] = ['mu', 'std', 'min', 'max']
    vector = vector.set_index('data')
    vector.loc['mu'] = [5, 8, 1]
    vector.loc['std'] = 20
    vector.loc['min'] = 0
    vector.loc['max'] = 100

    EDA = EDAc(MAX_ITER=20, DEAD_ITER=10, SIZE_GEN=40, ALPHA=0.6, vector=vector, cost_function=cost_function,
               aim='minimize')

    individual = {'param1': 5.1, 'param2': 1.0, 'param3': 6.9}
    assert EDA.__check_individual__(individual) == 13.0, "Should be 13.0"


def test_check_generation():
    # assert sum((1, 2, 2)) == 6, "Should be 6"
    from EDAspy.optimization.univariate.continuous import UMDAc as EDAc
    import pandas as pd

    def cost_function(dictionary):
        function = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
        if function < 0:
            return 9999999
        return function

    vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
    vector['data'] = ['mu', 'std', 'min', 'max']
    vector = vector.set_index('data')
    vector.loc['mu'] = [5, 8, 1]
    vector.loc['std'] = 20
    vector.loc['min'] = 0
    vector.loc['max'] = 100

    EDA = EDAc(MAX_ITER=20, DEAD_ITER=10, SIZE_GEN=8, ALPHA=0.6, vector=vector, cost_function=cost_function,
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


def test_individuals_selection():
    # assert sum((1, 2, 2)) == 6, "Should be 6"
    from EDAspy.optimization.univariate.continuous import UMDAc as EDAc
    import pandas as pd

    def cost_function(dictionary):
        function = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
        if function < 0:
            return 9999999
        return function

    vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
    vector['data'] = ['mu', 'std', 'min', 'max']
    vector = vector.set_index('data')
    vector.loc['mu'] = [5, 8, 1]
    vector.loc['std'] = 20
    vector.loc['min'] = 0
    vector.loc['max'] = 100

    EDA = EDAc(MAX_ITER=20, DEAD_ITER=10, SIZE_GEN=8, ALPHA=0.5, vector=vector, cost_function=cost_function,
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
    EDA.truncation()
    EDA.generation = EDA.generation.reset_index()
    del EDA.generation['index']

    output = expected_output == EDA.generation
    for col in list(expected_output.columns):
        assert list(output[col]) == [True, True, True, True], "Should be True"


def test_update_vector():
    # assert sum((1, 2, 2)) == 6, "Should be 6"
    from EDAspy.optimization.univariate.continuous import UMDAc as EDAc
    import pandas as pd

    def cost_function(dictionary):
        function = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
        if function < 0:
            return 9999999
        return function

    vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
    vector['data'] = ['mu', 'std', 'min', 'max']
    vector = vector.set_index('data')
    vector.loc['mu'] = [5, 8, 1]
    vector.loc['std'] = 20
    vector.loc['min'] = 0
    vector.loc['max'] = 100

    EDA = EDAc(MAX_ITER=20, DEAD_ITER=10, SIZE_GEN=8, ALPHA=0.5, vector=vector, cost_function=cost_function,
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

    EDA.generation = gen
    EDA.update_vector()

    expected_output = pd.DataFrame(columns=vector.columns)
    expected_output['data'] = ['mu', 'std', 'min', 'max']
    expected_output = expected_output.set_index('data')

    from scipy.stats import norm
    for col in list(expected_output.columns):
        mu, std = norm.fit(list(gen[col].values))
        expected_output.loc['mu', col] = mu
        if std < 1:
            std = 1
        expected_output.loc['std', col] = std

    expected_output.loc['min'] = 0
    expected_output.loc['max'] = 100

    output = expected_output == EDA.vector

    for col in list(vector.columns):
        assert list(output[col]) == [True, True, True, True], "Should be True"


if __name__ == "__main__":
    test__compare_costs__()
    test_check_individual_()
    test_check_generation()
    test_individuals_selection()
    test_update_vector()

    print("Everything passed")
