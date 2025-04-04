import numpy as np
import pytest
from src.c_class import CFunctionWrapper  # Adjust the import based on your project structure

# Fixture to create the wrapper once
@pytest.fixture(scope="module")
def wrapper():
    return CFunctionWrapper("src/c_class/C_functions.so")

def test_approximate_mass_left_hand_1(wrapper):
    """This will test the approximate left hand of a PDE discretised list. WIll test that the instance is a np.ndarray"""
    SSA_M = 10
    PDE_multiple = 4
    PDE_list = np.ones(SSA_M*PDE_multiple).astype(np.float32)
    deltax = 0.1

    result = wrapper.approximate_mass_left_hand(SSA_M, PDE_multiple, PDE_list, deltax)

    assert isinstance(result, np.ndarray)
   

def test_approximate_mass_left_hand_2(wrapper):
    """This will test the approximate left hand of a PDE discretised list"""
    SSA_M = 10
    PDE_multiple = 4
    PDE_list = np.ones(SSA_M*PDE_multiple).astype(np.float32)
    deltax = 0.1

    result = wrapper.approximate_mass_left_hand(SSA_M, PDE_multiple, PDE_list, deltax)

    assert len(result) == SSA_M
    
def test_approximate_mass_left_hand_3(wrapper):
    """This will test the approximate left hand of a PDE discretised list"""
    SSA_M = 10
    PDE_multiple = 4
    PDE_list = np.ones(SSA_M*PDE_multiple).astype(np.float32)
    deltax = 0.1

    result = wrapper.approximate_mass_left_hand(SSA_M, PDE_multiple, PDE_list, deltax)
    
    assert result[0] == 0.4 #This should be 1*0.1+1*0.1+1*0.1+1*0.1

def test_approximate_mass_left_hand_4(wrapper):
    """This will test the approximate left hand of a PDE discretised list"""
    SSA_M = 10
    PDE_multiple = 10
    PDE_list = np.ones(SSA_M*PDE_multiple).astype(np.float32)
    deltax = 0.1

    result = wrapper.approximate_mass_left_hand(SSA_M, PDE_multiple, PDE_list, deltax)
    
    assert all(i == 1 for i in result)

