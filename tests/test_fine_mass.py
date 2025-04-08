import numpy as np
import pytest
from src.c_class import CFunctionWrapper  # Adjust the import based on your project structure

# Fixture to create the wrapper once
@pytest.fixture(scope="module")
def test_data():
    """Provide consistent test inputs and params."""
    params = {
        "SSA_M": 3,
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


def test_fine_mass_1(wrapper):
    """
    Test the `fine_grid_ssa_mass` method to ensure the output is of the correct dtype.

    This test checks that the method returns a NumPy array.
    """
    SSA_mass = np.array([10, 20, 30], dtype=np.int32)  # 3 SSA compartments

    # Call the method
    fine_mass_output = wrapper.fine_grid_ssa_mass(SSA_mass)

    # Assert the output is a NumPy array
    assert isinstance(fine_mass_output, np.ndarray)


def test_fine_mass_2(wrapper):
    """
    Test the `fine_grid_ssa_mass` method to ensure the output has the correct length.

    This test checks that the length of the output matches the expected PDE grid length.
    """
    SSA_mass = np.array([10, 20, 30], dtype=np.int32)  # 3 SSA compartments
    SSA_M = wrapper.SSA_M
    PDE_multiple = wrapper.PDE_multiple

    # Call the method
    fine_mass_output = wrapper.fine_grid_ssa_mass(SSA_mass)

    # Assert the output length matches SSA_M * PDE_multiple
    assert len(fine_mass_output) == SSA_M * PDE_multiple


def test_fine_mass_3(wrapper):
    """
    Test the `fine_grid_ssa_mass` method for correct computation of the fine grid mass.

    This test checks that the output matches the expected fine grid mass distribution.
    """
    SSA_mass = np.array([10, 20, 30], dtype=np.int32)  # 3 SSA compartments
    SSA_M = wrapper.SSA_M
    PDE_multiple = wrapper.PDE_multiple
    h = wrapper.h

    # Call the method
    fine_mass_output = wrapper.fine_grid_ssa_mass(SSA_mass)

    # Expected fine mass output
    fine_mass_query = np.array(
        [10 / h] * PDE_multiple +
        [20 / h] * PDE_multiple +
        [30 / h] * PDE_multiple,
        dtype=np.float32
    )

    # Assert the output matches the expected fine grid mass
    assert np.array_equal(fine_mass_output, fine_mass_query)