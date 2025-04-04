import numpy as np
import pytest
from src.c_class import CFunctionWrapper  # Adjust the import based on your project structure

# Fixture to create the wrapper once
@pytest.fixture(scope="module")
def wrapper():
    return CFunctionWrapper("src/c_class/C_functions.so")

## Now we are going to test the  calculate_total_mass(self, PDE_list, SSA_list, PDE_multiple, deltax, SSA_M) function.

def test_calculate_total_mass_1(wrapper):
    """Ensure calculate_total_mass returns two NumPy arrays"""
    SSA_M = 2
    PDE_multiple = 2
    deltax = 0.1
    PDE_list = [1, 2, 3, 4]
    SSA_list = [5, 6]

    combined, approx_mass = wrapper.calculate_total_mass(PDE_list, SSA_list, PDE_multiple, deltax, SSA_M)

    assert isinstance(combined, np.ndarray)
    assert isinstance(approx_mass, np.ndarray)


def test_calculate_total_mass_2(wrapper):
    """Ensure calculate_total_mass returns two NumPy arrays"""
    SSA_M = 2
    PDE_multiple = 4
    deltax = 0.25
    PDE_list = [1, 2, 3, 4, 5, 6, 7, 8]  #(1+2+3+4)/4=2.5, (5+6+7+8)/4=6.5 Then we add them to 5 and 6.
    SSA_list = [5, 6]

    combined, approx_mass = wrapper.calculate_total_mass(PDE_list, SSA_list, PDE_multiple, deltax, SSA_M)

    combined_query = np.array([7.5, 12.5], dtype=np.float32) #Should be this asnwer
    assert np.array_equal(combined, combined_query)



