import numpy as np
import pytest
from src.c_class import CFunctionWrapper  # Adjust this import as needed

@pytest.fixture(scope="module")
def wrapper():
    return CFunctionWrapper("src/c_class/C_functions.so")

@pytest.fixture
def test_data():
    # Provide consistent test inputs
    SSA_M = 5
    PDE_list = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    SSA_list = np.array([1, 0, 2, 1, 3], dtype=np.int32)
    degradation_rate_h = 0.1
    threshold = 0.5
    production_rate = 1.0
    gamma = 2.0
    jump_rate = 0.05
    return {
        "SSA_M": SSA_M,
        "PDE_list": PDE_list,
        "SSA_list": SSA_list,
        "degradation_rate_h": degradation_rate_h,
        "threshold": threshold,
        "production_rate": production_rate,
        "gamma": gamma,
        "jump_rate": jump_rate,
    }

def test_propensity_output_keys(wrapper, test_data):
    """Check output contains all expected keys."""
    result = wrapper.calculate_propensity(**test_data)

    expected_keys = [
        "propensity_list",
        "boolean_SSA_list",
        "combined_mass_list",
        "approximate_PDE_mass",
        "boolean_mass_list"
    ]

    for key in expected_keys:
        assert key in result, f"Missing key in output: {key}"

def test_propensity_output_shapes_and_types(wrapper, test_data):
    """Check output arrays have correct shape and dtype."""
    result = wrapper.calculate_propensity(**test_data)
    SSA_M = test_data["SSA_M"]

    assert result["propensity_list"].shape == (SSA_M,), "Incorrect shape for 'propensity_list'"
    assert result["boolean_SSA_list"].shape == (SSA_M,), "Incorrect shape for 'boolean_SSA_list'"
    assert result["combined_mass_list"].shape == (SSA_M,), "Incorrect shape for 'combined_mass_list'"
    assert result["approximate_PDE_mass"].shape == (SSA_M,), "Incorrect shape for 'approximate_PDE_mass'"
    assert result["boolean_mass_list"].shape == (SSA_M,), "Incorrect shape for 'boolean_mass_list'"

    assert result["propensity_list"].dtype == np.float32, "Wrong dtype for 'propensity_list'"
    assert result["boolean_SSA_list"].dtype == np.int32, "Wrong dtype for 'boolean_SSA_list'"
    assert result["combined_mass_list"].dtype == np.float32, "Wrong dtype for 'combined_mass_list'"
    assert result["approximate_PDE_mass"].dtype == np.float32, "Wrong dtype for 'approximate_PDE_mass'"
    assert result["boolean_mass_list"].dtype == np.int32, "Wrong dtype for 'boolean_mass_list'"


def test_propensity_numerical(wrapper,test_data):
    """This will actually test the correct output"""
    result = wrapper.calculate_propensity(**test_data)

    propensity_list = result["propensity_list"]