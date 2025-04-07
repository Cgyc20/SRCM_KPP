import numpy as np
import pytest
from src.c_class import CFunctionWrapper  # Adjust the import based on your project structure

# Fixture to create the wrapper once
@pytest.fixture(scope="module")
def wrapper():
    return CFunctionWrapper("src/c_class/C_functions.so")



def test_fine_mass_1(wrapper):
    """Test number 1 is gonna be checking inputs and outputs is correct dtype."""

    SSA_mass = [10, 20, 30]        # 3 SSA compartments
    SSA_m = 3                      # number of SSA compartments
    PDE_multiple = 4               # each SSA maps to 4 PDE points
    PDE_grid_length = SSA_m * PDE_multiple
    h = 1.0                        # grid spacing

    # Run the function
    fine_mass_output = wrapper.fine_grid_ssa_mass(SSA_mass, PDE_grid_length, SSA_m, PDE_multiple, h)

    assert isinstance(fine_mass_output,np.ndarray)


def test_fine_mass_2(wrapper):
    """Tests if correct length wanted"""
    SSA_mass = [10, 20, 30]        # 3 SSA compartments
    SSA_m = 3                      # number of SSA compartments
    PDE_multiple = 4               # each SSA maps to 4 PDE points
    PDE_grid_length = SSA_m * PDE_multiple
    h = 1.0                        # grid spacing

    # Run the function
    fine_mass_output = wrapper.fine_grid_ssa_mass(SSA_mass, PDE_grid_length, SSA_m, PDE_multiple, h)

    assert len(fine_mass_output)==SSA_m*PDE_multiple #Test is correct length 


def test_fine_mass_3(wrapper):
    """Test if the output is correct"""


    SSA_mass = [10, 20, 30]        # 3 SSA compartments
    SSA_m = 3                      # number of SSA compartments
    PDE_multiple = 4               # each SSA maps to 4 PDE points
    PDE_grid_length = SSA_m * PDE_multiple
    h = 0.1                        # grid spacing

    # Run the function
    fine_mass_output = wrapper.fine_grid_ssa_mass(SSA_mass, PDE_grid_length, SSA_m, PDE_multiple, h)
    #The wquery fine mass is going to be the following:
    fine_mass_query = np.array([100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300], dtype=np.float32) #This is the expected output

    assert np.array_equal(fine_mass_output, fine_mass_query) #Check if the output is correct


