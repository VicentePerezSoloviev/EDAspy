from unittest import TestCase


class TestEDA_multivariate_gaussian(TestCase):
    def test_truncation(self):
        import pandas as pd
        from EDAspy.optimization.multivariate import EDA_multivariate_gaussian as EDAmg

        def cost_function(dictionary):
            suma = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
            if suma < 0:
                return 999999999
            return suma

        mus = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        mus.loc[0] = [10, 8, 5]

        sigma = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        sigma.loc[0] = 5

        EDA = EDAmg(SIZE_GEN=8, MAX_ITER=20, DEAD_ITER=10, ALPHA=0.5, aim='maximize',
                    cost_function=cost_function, mus=mus, sigma=sigma)

        gen = pd.DataFrame(columns=sigma.columns)

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

        expected_output = pd.DataFrame(columns=sigma.columns)

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
        print("EDAspy.optimization.multivariate.EDA_multivariate.truncation test passed")

    def test_check_generation(self):
        import pandas as pd
        from EDAspy.optimization.multivariate import EDA_multivariate_gaussian as EDAmg

        def cost_function(dictionary):
            suma = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
            if suma < 0:
                return 999999999
            return suma

        mus = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        mus.loc[0] = [10, 8, 5]

        sigma = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        sigma.loc[0] = 5

        EDA = EDAmg(SIZE_GEN=8, MAX_ITER=20, DEAD_ITER=10, ALPHA=0.5, aim='maximize',
                    cost_function=cost_function, mus=mus, sigma=sigma)

        gen = pd.DataFrame(columns=sigma.columns)

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
        print("EDAspy.optimization.multivariate.EDA_multivariate.check_generation test passed")

    def test_update_vector(self):
        import pandas as pd
        from EDAspy.optimization.multivariate import EDA_multivariate_gaussian as EDAmg

        def cost_function(dictionary):
            suma = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
            if suma < 0:
                return 999999999
            return suma

        mus = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        mus.loc[0] = [10, 8, 5]

        sigma = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        sigma.loc[0] = 5

        EDA = EDAmg(SIZE_GEN=8, MAX_ITER=20, DEAD_ITER=10, ALPHA=0.5, aim='maximize',
                    cost_function=cost_function, mus=mus, sigma=sigma)

        gen = pd.DataFrame(columns=sigma.columns)

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

        expected_sigma = pd.DataFrame(columns=sigma.columns)
        expected_sigma['vars'] = sigma.columns
        expected_sigma = expected_sigma.set_index('vars')
        expected_sigma.loc['param1'] = [1.0000, -0.0833, -0.0833]
        expected_sigma.loc['param2'] = [-0.0833, 1.0000, -0.0833]
        expected_sigma.loc['param3'] = [-0.0833, -0.0833, 1.0000]

        for var in list(sigma.columns):
            for varj in list(sigma.columns):
                # print(float(expected_sigma.round(decimals=4).loc[var, var]),
                # float(EDA.sigma.round(decimals=4).loc[var, var]))

                assert float(expected_sigma.round(decimals=4).loc[var, varj]) == \
                       float(EDA.sigma.round(decimals=4).loc[var, varj]), "Should be True"

        expected_mus = pd.DataFrame(columns=sigma.columns)
        expected_mus.loc[0] = 0.75

        for var in list(sigma.columns):
            assert float(expected_mus[var]) == float(EDA.mus[var]), "Should be True"
        print("EDAspy.optimization.multivariate.EDA_multivariate.update_vector test passed")

    def test_compare_costs(self):
        import pandas as pd
        from EDAspy.optimization.multivariate import EDA_multivariate_gaussian as EDAmg

        def cost_function(dictionary):
            suma = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
            if suma < 0:
                return 999999999
            return suma

        mus = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        mus.loc[0] = [10, 8, 5]

        sigma = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        sigma.loc[0] = 5

        EDA = EDAmg(SIZE_GEN=8, MAX_ITER=20, DEAD_ITER=10, ALPHA=0.5, aim='minimize',
                    cost_function=cost_function, mus=mus, sigma=sigma)

        EDA.best_mae_global = 10
        EDA.__compare_costs__(8)

        assert EDA.__compare_costs__(8) is True, "Should be True"
        print("EDAspy.optimization.multivariate.EDA_multivariate.__compare_costs__ test passed")

    def test_check_individual(self):
        import pandas as pd
        from EDAspy.optimization.multivariate import EDA_multivariate_gaussian as EDAmg

        def cost_function(dictionary):
            suma = dictionary['param1'] + dictionary['param2'] + dictionary['param3']
            if suma < 0:
                return 999999999
            return suma

        mus = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        mus.loc[0] = [10, 8, 5]

        sigma = pd.DataFrame(columns=['param1', 'param2', 'param3'])
        sigma.loc[0] = 5

        EDA = EDAmg(SIZE_GEN=8, MAX_ITER=20, DEAD_ITER=10, ALPHA=0.5, aim='minimize',
                    cost_function=cost_function, mus=mus, sigma=sigma)

        individual = {'param1': 5.1, 'param2': 1.0, 'param3': 6.9}
        assert EDA.__check_individual__(individual) == 13.0, "Should be 13.0"
        print("EDAspy.optimization.multivariate.EDA_multivariate._check_individual_ test passed")


