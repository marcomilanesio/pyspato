import unittest
import numpy as np
import utils

class TestUtilsMethods(unittest.TestCase):

    def test_standardize(self):
        x = np.linspace(0, 1000, 500)
        standardized = utils.standardize(x)
        assert np.allclose(np.mean(standardized), 0)
        assert np.allclose(np.std(standardized), 1)


if __name__ == '__main__':
    unittest.main()

