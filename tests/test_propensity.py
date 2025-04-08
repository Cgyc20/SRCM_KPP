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
    PDE_list = np.array([1.2, 1.4, 0.4, 0.5, 0.6, 0.8, 0.1, 0.8, 1.0, 1.2], dtype=np.float32)
    SSA_list = np.array([100, 1, 2, 1, 3], dtype=np.int32)
    degradation_rate_h = 0.1
    PDE_multiple = 2
    threshold = 0.5
    production_rate = 1.0
    gamma = 2.0
    jump_rate = 0.05
    h = 0.1
    deltax = 0.1
    return {
        "SSA_M": SSA_M,
        "PDE_multiple": PDE_multiple,
        "PDE_list": PDE_list,
        "SSA_list": SSA_list,
        "degradation_rate_h": degradation_rate_h,
        "threshold": threshold,
        "production_rate": production_rate,
        "gamma": gamma,
        "jump_rate": jump_rate,
        "h": h,
        "deltax": deltax,
    }

def test_propensity_output_keys(wrapper, test_data):
    """Check output contains all expected keys."""
    result = wrapper.calculate_propensity(**test_data)

    expected_keys = [
        "propensity_list",
        "boolean_SSA_list",
        "boolean_PDE_list",
        "combined_mass_list",
        "approximate_PDE_mass",
        "boolean_mass_list",
    ]

    for key in expected_keys:
        assert key in result, f"Missing key in output: {key}"

def test_propensity_PDE_bool(wrapper,test_data):
    """Check boolean_PDE_list contains only 0s and 1s."""
    result = wrapper.calculate_propensity(**test_data)
    boolean_PDE_list = result["boolean_PDE_list"]
    #print(f"PDE bool list: {boolean_PDE_list}")
    assert np.all(np.isin(boolean_PDE_list, [0, 1])), "boolean_PDE_list should contain only 0s and 1s"


def test_propensity_boolean_SSA_list(wrapper, test_data):
    """Check boolean_SSA_list contains only 0s and 1s."""
    result = wrapper.calculate_propensity(**test_data)
    boolean_SSA_list = result["boolean_SSA_list"]
    #print(boolean_SSA_list)
    assert np.all(np.isin(boolean_SSA_list, [0, 1])), "boolean_SSA_list should contain only 0s and 1s"

def test_propensity_numerical(wrapper, test_data):
    """Test with smaller steps to isolate segfault"""
    # Step 1: Just call the function
    result = wrapper.calculate_propensity(**test_data)
    #print("Function call completed")  # Check if this prints
    
    # Step 2: Access propensity_list
    propensity_list = result["propensity_list"]
    #print("Accessed propensity_list")  # Check if this prints
    
    # Step 3: Verify content
    #print(propensity_list[:10])  # Check first 10 elements



def test_propensity_output_shapes_and_types(wrapper, test_data):
    """Check output arrays have correct shape and dtype."""
    result = wrapper.calculate_propensity(**test_data)
    #print(result["propensity_list"])
    #print(len(result["propensity_list"]))
    SSA_M = test_data["SSA_M"]
    #print(f"propensity_shape = {result["propensity_list"].shape}")
    assert result["propensity_list"].shape == (6*SSA_M,), "Incorrect shape for 'propensity_list'"
    #print(result["boolean_SSA_list"])
    assert result["boolean_SSA_list"].shape == (SSA_M,), "Incorrect shape for 'boolean_SSA_list'"
    
    assert result["combined_mass_list"].shape == (SSA_M,), "Incorrect shape for 'combined_mass_list'"
    assert result["approximate_PDE_mass"].shape == (SSA_M,), "Incorrect shape for 'approximate_PDE_mass'"
    assert result["boolean_mass_list"].shape == (SSA_M,), "Incorrect shape for 'boolean_mass_list'"

    assert result["propensity_list"].dtype == np.float32, "Wrong dtype for 'propensity_list'"
    assert result["boolean_SSA_list"].dtype == np.int32, "Wrong dtype for 'boolean_SSA_list'"
    assert result["combined_mass_list"].dtype == np.float32, "Wrong dtype for 'combined_mass_list'"
    #print(f"combined list datatype : {result["combined_mass_list"].dtype}")
    assert result["approximate_PDE_mass"].dtype == np.float32, "Wrong dtype for 'approximate_PDE_mass'"
    assert result["boolean_mass_list"].dtype == np.int32, "Wrong dtype for 'boolean_mass_list'"


def test_propensity_numerical(wrapper,test_data):
    """This will actually test the correct output"""
    result = wrapper.calculate_propensity(**test_data)

    propensity_list = result["propensity_list"]
    boolean_SSA_mass_list = result["boolean_SSA_list"]
    combined_mass_list = result["combined_mass_list"]
    approximate_PDE_mass = result["approximate_PDE_mass"]
    boolean_mass_list = result["boolean_mass_list"]

    # print(f"propensity_list: {propensity_list}")
    # print(f"boolean_SSA_mass_list: {boolean_SSA_mass_list}")
    # print(f"combined_mass_list: {combined_mass_list}")
    # print(f"approximate_PDE_mass: {approximate_PDE_mass}")
    # print(f"boolean_mass_list: {boolean_mass_list}")
    # #Now print dtypes
    # print(f"propensity_list dtype: {propensity_list.dtype}")
    # print(f"boolean_SSA_mass_list dtype: {boolean_SSA_mass_list.dtype}")
    # print(f"combined_mass_list dtype: {combined_mass_list.dtype}")
    # print(f"approximate_PDE_mass dtype: {approximate_PDE_mass.dtype}")
    # print(f"boolean_mass_list dtype: {boolean_mass_list.dtype}")

    SSA_M = test_data["SSA_M"]
    assert propensity_list.shape == (6*SSA_M,)