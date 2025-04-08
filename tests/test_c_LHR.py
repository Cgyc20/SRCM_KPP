import numpy as np
import pytest
from src.c_class import CFunctionWrapper  # Adjust the import based on your project structure

# Fixture to create the wrapper once
@pytest.fixture(scope="module")
def test_data():
    """Provide consistent test inputs and params."""
    params = {
        "SSA_M": 10,
        "PDE_multiple": 4,
        "deltax": 0.1,
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


def test_approximate_mass_left_hand_1(wrapper):
    """
    Test the `approximate_mass_left_hand` method to ensure it returns a NumPy array.

    This test checks that the result is an instance of `np.ndarray` when given a valid PDE list.
    """
    SSA_M = wrapper.SSA_M
    PDE_multiple = wrapper.PDE_multiple

    # Create a PDE list with ones
    PDE_list = np.ones(SSA_M * PDE_multiple, dtype=np.float32)

    # Call the method
    result = wrapper.approximate_mass_left_hand(PDE_list)

    # Assert the result is a NumPy array
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == SSA_M


def test_approximate_mass_left_hand_2(wrapper):
    """
    Test the `approximate_mass_left_hand` method to ensure the result length matches SSA_M.
    """
    SSA_M = wrapper.SSA_M
    PDE_multiple = wrapper.PDE_multiple

    # Create a PDE list with ones
    PDE_list = np.ones(SSA_M * PDE_multiple, dtype=np.float32)

    # Call the method
    result = wrapper.approximate_mass_left_hand(PDE_list)

    # Assert the result length matches SSA_M
    assert len(result) == SSA_M


def test_approximate_mass_left_hand_3(wrapper):
    """
    Test the `approximate_mass_left_hand` method for correct computation of the first value.
    """
    SSA_M = wrapper.SSA_M
    PDE_multiple = wrapper.PDE_multiple
    deltax = wrapper.deltax

    # Create a PDE list with ones
    PDE_list = np.ones(SSA_M * PDE_multiple, dtype=np.float32)

    # Call the method
    result = wrapper.approximate_mass_left_hand(PDE_list)

    # Assert the first value is computed correctly
    expected_value = PDE_multiple * deltax  # Sum of PDE_multiple * deltax
    assert result[0] == expected_value


def test_approximate_mass_left_hand_4(wrapper):
    """
    Test the `approximate_mass_left_hand` method for uniform PDE values.

    This ensures that all values in the result are equal when the PDE list is uniform.
    """
    SSA_M = wrapper.SSA_M
    PDE_multiple = wrapper.PDE_multiple
    deltax = wrapper.deltax

    # Create a PDE list with ones
    PDE_list = np.ones(SSA_M * PDE_multiple, dtype=np.float32)

    # Call the method
    result = wrapper.approximate_mass_left_hand(PDE_list)

    # Assert all values in the result are equal to PDE_multiple * deltax
    expected_value = PDE_multiple * deltax
    assert all(value == expected_value for value in result)
