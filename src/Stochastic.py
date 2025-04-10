import numpy as np
from tqdm import tqdm
import os
import json
from copy import deepcopy, copy
import ctypes
from .c_class import SSA_C_Wrapper

class SSA:

    def __init__(self,input):
        # Initialise the parameters from input dictionary


        self.L = input['domain_length']
        self.SSA_M = input['compartment_number']
        self.total_time = input['total_time']
        self.timestep = input['timestep']
        self.degradation_rate = input['degradation_rate']
        self.production_rate = input['production_rate']
        self.diffusion_rate = input['diffusion_rate']
        self.h = input['h']
        self.d = self.diffusion_rate / (self.h**2)

          # Validate SSA_initial
        SSA_initial = input.get('SSA_initial')
        if not isinstance(SSA_initial, np.ndarray):
            raise ValueError("SSA_initial must be a numpy array.")
        if len(SSA_initial) != self.SSA_M:
            raise ValueError("The length of SSA_initial must match the number of compartments (SSA_M).")
        if not np.issubdtype(SSA_initial.dtype, np.integer):
            raise ValueError("SSA_initial must contain integer values.")
        self.SSA_initial = SSA_initial.astype(int)

        params_dict = {
            'domain_length': self.L,
            'compartment_number': self.SSA_M,
            'total_time': self.total_time,
            'timestep': self.timestep,
            'degradation_rate': self.degradation_rate,
            'diffusion_rate': self.diffusion_rate,
            'production_rate':self.production_rate,
            'h': self.h,
        }

        self.C_wrapper = SSA_C_Wrapper(params_dict, library_path="src/c_class/C_functions.so")

        self.time_vector = np.arange(0, self.total_time, self.timestep)
        self.SSA_X = np.linspace(0, self.L - self.h, self.SSA_M)

    def create_initial_dataframe(self):
        """Creates the initial dataframes to be used throughout the simulation
        Returns: the initial dataframes for discrete and continuous numbers of molecules. With initial conditions"""

        SSA_matrix = np.zeros((self.SSA_M, len(self.time_vector)))  # Discrete molecules grid
        SSA_matrix[:, 0] = self.SSA_initial
        return SSA_matrix


    def stochastic_simulation(self, SSA_grid):
        t = 0
        old_time = t
        SSA_list = deepcopy(SSA_grid[:, 0])  # Starting SSA_list
        while t < self.total_time:
            dataframe = self.C_wrapper.calculate_propensity(SSA_list)  # Calculate the propensity functions
            total_propensity = dataframe["propensity"]
            alpha0 = np.sum(total_propensity)
            if alpha0 == 0:  # Stop if no reactions can occur
                break
         
            r1, r2, r3 = np.random.rand(3)
            tau = (1 / alpha0) * np.log(1 / r1)  # Time until next reaction
            alpha_cum = np.cumsum(total_propensity)  # Cumulative sum of propensities
            index = np.searchsorted(alpha_cum, r2 * alpha0)  # Determine which reaction occurs

            compartment_index = index % self.SSA_M  # The compartmental index is just the modulo of SSA.
            if index <= self.SSA_M - 2 and index >= 1:
                if r3 < 0.5:  # Move left
                    SSA_list[index] = SSA_list[index] - 1
                    SSA_list[index - 1] += 1
                else:  # Move right
                    SSA_list[index] = SSA_list[index] - 1
                    SSA_list[index + 1] += 1
            elif index == 0:  # Left boundary (can only move right)
                SSA_list[index] = SSA_list[index] - 1
                SSA_list[index + 1] += 1
            elif index == self.SSA_M - 1:  # Right boundary (can only move left)
                SSA_list[index] = SSA_list[index] - 1
                SSA_list[index - 1] += 1
            elif index >= self.SSA_M and index <= 2 * self.SSA_M - 1:  # Production reaction
                SSA_list[compartment_index] += 1
            elif index >= 2 * self.SSA_M and index <= 3 * self.SSA_M - 1:  # Degradation reaction
                SSA_list[compartment_index] -= 1
            ind_before = np.searchsorted(self.time_vector, old_time, 'right')
            ind_after = np.searchsorted(self.time_vector, t, 'left')
            for time_index in range(ind_before, min(ind_after + 1, len(self.time_vector))):
                SSA_grid[:, time_index] = SSA_list

            old_time = t  # Update old_time
            t += tau  # Update time by the time step

        return SSA_grid

    def run_simulation(self, number_of_repeats):
        """This will run the simulation with a total number of repeats"""
        SSA_initial = self.create_initial_dataframe()
        SSA_average = np.zeros_like(SSA_initial)
        filled_SSA_grid = np.zeros_like(SSA_initial)

        for _ in tqdm(range(number_of_repeats), desc="Running the Stochastic simulations"):
            SSA_grid_initial = self.create_initial_dataframe()
            SSA_current = self.stochastic_simulation(SSA_grid_initial)
            SSA_average += SSA_current

        filled_SSA_grid = SSA_average / number_of_repeats

        print("Simulation completed")
        return filled_SSA_grid

    def save_simulation_data(self, filled_SSA_grid, datadirectory='data'):
        if not os.path.exists(datadirectory):
            os.makedirs(datadirectory)

        params = {
            'domain_length': self.L,
            'compartment_number': self.SSA_M,
            'total_time': self.total_time,
            'timestep': self.timestep,
            'production_rate': self.production_rate,
            'degradation_rate': self.degradation_rate,
            'diffusion_rate': self.diffusion_rate,
            'initial_SSA': self.SSA_initial.tolist(),
            'h': self.h,
        }

        np.savez(os.path.join(datadirectory, 'Pure_SSA_data'),
                 SSA_grid=filled_SSA_grid,
                 time_vector=self.time_vector,
                 SSA_X=self.SSA_X
                 )

        with open(os.path.join(datadirectory, "parameters_pure_SSA.json"), 'w') as params_file:
            json.dump(params, params_file, indent=4)

        print("Data saved successfully")


