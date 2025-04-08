import numpy as np
import pytest
from src.c_class import CFunctionWrapper  # Adjust the import based on your project structure

# Fixture to create the wrapper once
@pytest.fixture(scope="module")
def test_data():
    """Provide consistent test inputs and params."""
    params = {
        "SSA_M": 2,
        "PDE_multiple": 2,
        "deltax": 0.25,
        "h": 0.1,
        "threshold": 0.5,
        "production_rate": 1.0,
        "degradation_rate_h": 0.1,
        "jump_rate": 0.05,
        "gamma": 2.0
    }
    return {"params": params}

@pytest.fixture(scope="module")
def wrapper(test_data):
    """Create the CFunctionWrapper instance using the provided params."""
    return CFunctionWrapper(params=test_data["params"], library_path="src/c_class/C_functions.so")


def test_calculate_total_mass_1(wrapper):
    """
    Test the `calculate_total_mass` method to ensure it returns two NumPy arrays.

    This test checks that the method outputs are of the correct type.
    """
    # Test data
    PDE_list = np.array([1, 2, 3, 4], dtype=np.float32)
    SSA_list = np.array([5, 6], dtype=np.int32)

    # Call the method
    combined, approx_mass = wrapper.calculate_total_mass(PDE_list, SSA_list)

    # Assert the outputs are NumPy arrays
    assert isinstance(combined, np.ndarray)
    assert isinstance(approx_mass, np.ndarray)


def test_calculate_total_mass_2(wrapper):
    """Ensure calculate_total_mass returns two NumPy arrays"""
    SSA_M = 2
    PDE_multiple = 4
    deltax = 0.25
    PDE_list = [1, 2, 3, 4]  #(1+2)*0.25=0.75, (3+4)*0.25= 1.75 Then we add them to 5 and 6.
    SSA_list = [5, 6]

    combined, approx_mass = wrapper.calculate_total_mass(PDE_list, SSA_list)
    print(combined)
    combined_query = np.array([5.75, 7.75], dtype=np.float32) #Should be this asnwer
    assert np.array_equal(combined, combined_query)



