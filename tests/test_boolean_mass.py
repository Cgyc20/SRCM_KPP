import numpy as np
import pytest
from src.c_class import CFunctionWrapper  # Adjust the import based on your project structure

# Fixture to create the wrapper once
@pytest.fixture(scope="module")
def test_data():
    """Provide consistent test inputs and params."""
    
    PDE_list = np.array([1.2, 1.4, 0.4, 0.5, 0.6, 0.8, 0.1, 0.8, 11.0, 12.0], dtype=np.float32)
    SSA_list = np.array([100, 1, 2, 1, 3], dtype=np.int32)
    params = {
        "SSA_M": 3,
        "PDE_multiple": 2,
        "deltax": 0.1,
        "h": 0.1,
        "threshold": 0.5,
        "production_rate": 1.0,
        "degradation_rate_h": 0.1,
        "jump_rate": 0.05,
        "gamma": 2.0
    }

    return {
        "params": params,
    }

@pytest.fixture(scope="module")
def wrapper(test_data):
    """Create the CFunctionWrapper instance using the provided params."""
    return CFunctionWrapper(params=test_data["params"], library_path="src/c_class/C_functions.so")



def test_boolean_mass_1(wrapper):
    """Check that boolean_low_limit returns two NumPy arrays"""
   
    PDE_list = [0, 0, 1, 1, 0, 0]  # just dummy data

    boolean_PDE, boolean_SSA = wrapper.boolean_low_limit(PDE_list)

    assert isinstance(boolean_PDE, np.ndarray)
    assert isinstance(boolean_SSA, np.ndarray)


def test_boolean_mass_2(wrapper):
    """This will check whether the datatyype in integer32"""
   
    PDE_list = [0, 0, 1, 1, 0, 0]  # just dummy data

    boolean_PDE, boolean_SSA = wrapper.boolean_low_limit(PDE_list)

    assert boolean_PDE.dtype == np.int32
    assert boolean_SSA.dtype == np.int32


def test_boolean_mass_3(wrapper):
    """Ths will check whether the PDE boolean mass works"""

  
    PDE_list = [1, 20, 30, 5, 0, 0]  # just dummy data

    boolean_PDE, boolean_SSA = wrapper.boolean_low_limit(PDE_list)

    boolean_PDE_query = [0,1,1,0,0,0]
    boolean_SSA_query = [1,0]

    assert np.array_equal(boolean_PDE, boolean_PDE_query)

def test_boolean_mass_4(wrapper):
    """Ths will check whether the SSA boolean mass works"""

    PDE_list = [100, 20, 30, 5, 0, 0]  # just dummy data

    boolean_PDE, boolean_SSA = wrapper.boolean_low_limit(PDE_list)

    boolean_SSA_query = [1,0,0] #Since the PDE bool is [1,1,1,0,0,]. SO firs double are both 1

    assert np.array_equal(boolean_SSA, boolean_SSA_query)

