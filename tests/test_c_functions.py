import numpy as np
import pytest
from src.c_class.python_wrapper import CFunctionWrapper  # Adjust if needed

# Fixture to create the wrapper once
@pytest.fixture(scope="module")
def wrapper():
    return CFunctionWrapper("src/c_class/C_functions.so")

def test_approximate_mass_left_hand(wrapper):
    SSA_M = 10
    PDE_multiple = 2
    PDE_list = np.random.rand(20).astype(np.float32)
    deltax = 0.1

    result = wrapper.approximate_mass_left_hand(SSA_M, PDE_multiple, PDE_list, deltax)

    assert isinstance(result, np.ndarray)
    assert len(result) == SSA_M