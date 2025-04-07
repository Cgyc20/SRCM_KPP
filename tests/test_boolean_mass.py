import numpy as np
import pytest
from src.c_class import CFunctionWrapper  # Adjust the import based on your project structure

# Fixture to create the wrapper once
@pytest.fixture(scope="module")
def wrapper():
    return CFunctionWrapper("src/c_class/C_functions.so")



def test_boolean_mass_1(wrapper):
    """Check that boolean_mass returns two NumPy arrays"""
    SSA_M = 3
    PDE_M = 6
    PDE_multiple = 2
    h = 0.1
    PDE_list = [0, 0, 1, 1, 0, 0]  # just dummy data

    boolean_PDE, boolean_SSA = wrapper.boolean_mass(SSA_M, PDE_M, PDE_multiple, PDE_list, h)

    assert isinstance(boolean_PDE, np.ndarray)
    assert isinstance(boolean_SSA, np.ndarray)


def test_boolean_mass_2(wrapper):
    """This will check whether the datatyype in integer32"""
    SSA_M = 3
    PDE_M = 6
    PDE_multiple = 2
    h = 0.1
    PDE_list = [0, 0, 1, 1, 0, 0]  # just dummy data

    boolean_PDE, boolean_SSA = wrapper.boolean_mass(SSA_M, PDE_M, PDE_multiple, PDE_list, h)

    assert boolean_PDE.dtype == np.int32
    assert boolean_SSA.dtype == np.int32


def test_boolean_mass_3(wrapper):
    """Ths will check whether the PDE boolean mass works"""

    SSA_M = 3
    PDE_M = 6
    PDE_multiple = 2
    h = 0.1
    PDE_list = [1, 20, 30, 5, 0, 0]  # just dummy data

    boolean_PDE, boolean_SSA = wrapper.boolean_mass(SSA_M, PDE_M, PDE_multiple, PDE_list, h)

    boolean_PDE_query = [0,1,1,0,0,0]
    boolean_SSA_query = [1,0]

    assert np.array_equal(boolean_PDE, boolean_PDE_query)

def test_boolean_mass_4(wrapper):
    """Ths will check whether the SSA boolean mass works"""

    SSA_M = 3
    PDE_M = 6
    PDE_multiple = 2
    h = 0.1
    PDE_list = [100, 20, 30, 5, 0, 0]  # just dummy data

    boolean_PDE, boolean_SSA = wrapper.boolean_mass(SSA_M, PDE_M, PDE_multiple, PDE_list, h)

    boolean_SSA_query = [1,0,0] #Since the PDE bool is [1,1,1,0,0,]. SO firs double are both 1

    assert np.array_equal(boolean_SSA, boolean_SSA_query)

