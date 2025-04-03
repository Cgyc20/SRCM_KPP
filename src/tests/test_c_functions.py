import unittest
import numpy as np
from ..c_class import CFunctionWrapper


class TestCFunctionWrapper(unittest.TestCase):
    def setUp(self):
        self.wrapper = CFunctionWrapper()
        
    def test_approximate_mass_left_hand(self):
        SSA_M = 10
        PDE_multiple = 2
        PDE_list = np.random.rand(20).astype(np.float32)
        deltax = 0.1
        result = self.wrapper.approximate_mass_left_hand(SSA_M, PDE_multiple, PDE_list, deltax)
        self.assertEqual(len(result), SSA_M)

    # Add more tests for other methods...

if __name__ == "__main__":
    unittest.main()


    