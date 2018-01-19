import unittest
import numpy as np
import utils


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        size = 500
        self.x = np.linspace(0, 1000, size)

    def test_standardize(self):
        stdx = utils.standardize(self.x)
        assert np.allclose(np.mean(stdx), 0)
        assert np.allclose(np.std(stdx), 1)


if __name__ == '__main__':
    unittest.main()

