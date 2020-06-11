from unittest import TestCase


class Test1(TestCase):
    def test_normalizacion(self):
        from EDAspy.optimization.multivariate.__matrix import normalizacion
        totest = [0.20, 0.60, 0.2, 0.60]
        expected_output = [0.125, 0.375, 0.125, 0.375]

        assert normalizacion(totest) == expected_output, "Should be True"
        print("EDAspy.optimization.multivariate.__matrix.normalizacion test passed")


class Test2(TestCase):
    def test_is_pd(self):
        from EDAspy.optimization.multivariate.__matrix import isPD
        import numpy as np

        matrix1 = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        matrix3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        assert isPD(matrix1) == False, "Should be False"
        assert isPD(matrix3) == True, "Should be True"
        print("EDAspy.optimization.multivariate.__matrix.isPD test passed")


class Test3(TestCase):
    def test_nearest_pd(self):
        from EDAspy.optimization.multivariate.__matrix import nearestPD
        import numpy as np

        matrix1 = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        expected_output = np.array([[1.42710507, 2.71987509, 2.07349008],
                                    [2.71987509, 5.1837252, 3.95180015],
                                    [2.07349008, 3.95180015, 3.01264511]]).round(decimals=4)

        output = nearestPD(matrix1).round(decimals=4)

        assert output.all() == expected_output.all(), "Should be True"
        print("EDAspy.optimization.multivariate.__matrix.nearestPD test passed")


class Test4(TestCase):
    def test_is_invertible(self):
        from EDAspy.optimization.multivariate.__matrix import is_invertible
        import numpy as np

        matrix1 = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        matrix2 = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 4]])

        assert is_invertible(matrix1) is not True, "Should be False"
        assert is_invertible(matrix2) == True, "Should be True"
        print("EDAspy.optimization.multivariate.__matrix.is_invertible test passed")