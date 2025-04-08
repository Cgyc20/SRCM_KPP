import numpy as np
from tqdm import tqdm
import os
import json
from copy import deepcopy, copy
import ctypes
from .c_class import CFunctionWrapper, Numerical_wrapper


class Hybrid:

    def __init__(self, input):
        # Initialize parameters from input dictionary
        self.L = input['domain_length']
        self.SSA_M = input['compartment_number']
        self.PDE_multiple = input['PDE_multiple']
        self.total_time = input['total_time']
        self.timestep = input['timestep']
        self.threshold = input['threshold']
        self.gamma = input['gamma']
        self.degradation_rate = input['degradation_rate']
        self.diffusion_rate = input['diffusion_rate']
        self.h = input['h']
        self.d = self.diffusion_rate / (self.h**2)
        self.deltax = input['deltax']
        self.production_rate = input['production_rate']
        self.threshold_conc = input['threshold_conc']
        self.SSA_X = np.linspace(0, self.L - self.h, self.SSA_M)
        self.PDE_M = self.SSA_M * self.PDE_multiple
        self.PDE_X = np.linspace(0, self.L, self.PDE_M) 

    
        # Validate SSA_initial
        SSA_initial = input.get('SSA_initial')
        if not isinstance(SSA_initial, np.ndarray):
            raise ValueError("SSA_initial must be a numpy array.")
        if len(SSA_initial) != self.SSA_M:
            raise ValueError("The length of SSA_initial must match the number of compartments (SSA_M).")
        if not np.issubdtype(SSA_initial.dtype, np.integer):
            raise ValueError("SSA_initial must contain integer values.")
        self.SSA_initial = SSA_initial.astype(int)

        # Initialize additional attributes
        self.PDE_initial_conditions = np.zeros_like(self.PDE_X, dtype=np.float64)
        self.steady_state = self.production_rate / self.degradation_rate

        self.time_vector = np.arange(0, self.total_time, self.timestep)

        Cfunctions_params = {'SSA_M': self.SSA_M,
                            'PDE_multiple': self.PDE_multiple,
                            'PDE_M': self.PDE_M,
                            'h': self.h,
                            'deltax': self.deltax,
                            'threshold': self.threshold,
                            'production_rate':self.production_rate,
                            'degradation_rate_h': self.degradation_rate/self.h,
                            'jump_rate': self.d,
                            'gamma': self.gamma,
                            }

        Numerical_params = {'PDE_M': self.PDE_M,
                            'diffusion_rate': self.diffusion_rate,
                            'degradation_rate': self.degradation_rate,
                            'production_rate': self.production_rate,
                            'deltax': self.deltax,
                            'timestep': self.timestep,}
        
        self.CFunctions = CFunctionWrapper(Cfunctions_params,"src/c_class/C_functions.so")
    
        self.NumericalClass = Numerical_wrapper(Numerical_params)


        print("Successfully initialized the hybrid model")
        print(f"The threshold concentration is: {self.threshold_conc}")

    def create_initial_dataframe(self) -> np.ndarray:
        SSA_grid = np.zeros((self.SSA_M, len(self.time_vector)), dtype=int)
        SSA_grid[:, 0] = self.SSA_initial
        PDE_grid = np.zeros((self.PDE_M, len(self.time_vector)), dtype=float)
        PDE_grid[:, 0] = self.PDE_initial_conditions
        return PDE_grid, SSA_grid 

    

    def hybrid_simulation(self, SSA_grid: np.ndarray, PDE_grid: np.ndarray, approx_mass: np.ndarray) -> np.ndarray:
        t = 0
        old_time = t
        td = self.timestep
        PDE_particles = np.zeros_like(approx_mass)
        SSA_list = SSA_grid[:, 0].astype(int)
        PDE_list = PDE_grid[:, 0].astype(float)

        print(f"Debug: Length of PDE_list = {len(PDE_list)}, Expected PDE_length = {self.PDE_M}")
        ind_after = 0

        while t < self.total_time:
            # Use the correct instance for propensity calculation
            dataframe = self.CFunctions.calculate_propensity(PDE_list, SSA_list)
            fine_SSA_mass = self.CFunctions.fine_grid_ssa_mass(SSA_list)

            alpha0 = np.sum(dataframe["propensity_list"])
            if alpha0 == 0:
                PDE_list = self.NumericalClass.RK4_steps(PDE_list, dataframe["PDE_bool_list"], fine_SSA_mass)

                t = copy(td)
                td += self.timestep
                ind_before = np.searchsorted(self.time_vector, old_time, 'right')
                ind_after = np.searchsorted(self.time_vector, t, 'left')
                for time_index in range(ind_before, min(ind_after + 1, len(self.time_vector))):
                    PDE_grid[:, time_index] = PDE_list
                    SSA_grid[:, time_index] = SSA_list
                    self.check_negative_values(PDE_list, "PDE_list")
                    self.check_negative_values(SSA_list, "SSA_list")
                    approx_mass[:, time_index], PDE_particles[:, time_index] = self.CFunctions.calculate_total_mass(PDE_list, SSA_list)
                old_time = t
                continue

            r1, r2, r3 = np.random.rand(3)
            tau = (1 / alpha0) * np.log(1 / r1)
            alpha_cum = np.cumsum(dataframe["propensity_list"])
            index = np.searchsorted(alpha_cum, r2 * alpha0)
            compartment_index = index % self.SSA_M

            if t + tau <= td:
                reaction_type = ""
                if index <= self.SSA_M - 2 and index >= 1:  # Diffusion
                    if r3 < 0.5:
                        SSA_list[index] -= 1
                        SSA_list[index - 1] += 1
                        reaction_type = "diffusion"
                    else:
                        SSA_list[index] -= 1
                        SSA_list[index + 1] += 1
                        reaction_type = "diffusion"
                elif index == 0:
                    SSA_list[index] -= 1
                    SSA_list[index + 1] += 1
                    reaction_type = "diffusion"
                elif index == self.SSA_M - 1:
                    SSA_list[index] -= 1
                    SSA_list[index - 1] += 1
                    reaction_type = "diffusion"
                elif index >= self.SSA_M and index <= 2 * self.SSA_M - 1:
                    SSA_list[compartment_index] += 1  # D -> 2D
                    reaction_type = "D duplication"
                elif index >= 2 * self.SSA_M and index <= 3 * self.SSA_M - 1:
                    SSA_list[compartment_index] -= 1  # 2D -> D
                    reaction_type = "D degradation"
                elif index >= 3 * self.SSA_M and index <= 4 * self.SSA_M - 1:
                    SSA_list[compartment_index] -= 1  # D + C -> C
                    reaction_type = "J degredation"
                elif index >= 4 * self.SSA_M and index <= 5 * self.SSA_M - 1:  # C -> D
                    SSA_list[compartment_index] += 1
                    PDE_list[self.PDE_multiple * compartment_index: self.PDE_multiple * (compartment_index + 1)] -= 1 / self.h
                    reaction_type = "conversion_C_to_D"
                else:
                    SSA_list[compartment_index] -= 1  # D -> C
                    PDE_list[self.PDE_multiple * compartment_index: self.PDE_multiple * (compartment_index + 1)] += 1 / self.h
                    reaction_type = "conversion_D_to_C"

                t += tau
                old_time = t

            else:
                PDE_list = self.NumericalClass.RK4(PDE_list, dataframe["PDE_bool_list"], fine_SSA_mass)

                t = copy(td)
                td += self.timestep
                ind_before = np.searchsorted(self.time_vector, old_time, 'right')
                ind_after = np.searchsorted(self.time_vector, t, 'left')
                for time_index in range(ind_before, min(ind_after + 1, len(self.time_vector))):
                    PDE_grid[:, time_index] = PDE_list
                    SSA_grid[:, time_index] = SSA_list
                    self.check_negative_values(PDE_list, "PDE_list")
                    self.check_negative_values(SSA_list, "SSA_list")
                    approx_mass[:, time_index], PDE_particles[:, time_index] = self.CFunctions.calculate_total_mass(PDE_list, SSA_list)

                old_time = t

        return SSA_grid, PDE_grid, approx_mass

    def run_simulation(self, number_of_repeats: int) -> np.ndarray:
        PDE_initial, SSA_initial = self.create_initial_dataframe()
        approx_mass_initial = np.zeros_like(SSA_initial)
        approx_mass_initial[:, 0], _ = self.CFunctions.calculate_total_mass(PDE_initial[:, 0], SSA_initial[:, 0])
        SSA_sum = np.zeros_like(SSA_initial)
        PDE_sum = np.zeros_like(PDE_initial)
        approx_mass_sum = np.zeros_like(approx_mass_initial)

        # Arrays to track all SSA events and PDE update times across repeats
        all_SSA_events_logs = []
        all_PDE_update_times = []

        for _ in tqdm(range(number_of_repeats), desc="Running the Hybrid simulations"):
            SSA_current, PDE_current, approx_mass_current, SSA_events_log, PDE_update_times = self.hybrid_simulation(
                deepcopy(SSA_initial), deepcopy(PDE_initial), deepcopy(approx_mass_initial))
            SSA_sum += SSA_current
            PDE_sum += PDE_current
            approx_mass_sum += approx_mass_current

            all_SSA_events_logs.append(SSA_events_log)
            all_PDE_update_times.append(PDE_update_times)

        SSA_average = SSA_sum / number_of_repeats
        PDE_average = PDE_sum / number_of_repeats

        combined_grid = np.zeros_like(PDE_average)
        for i in range(SSA_average.shape[1]):
            for j in range(SSA_average.shape[0]):
                start_index = j * self.PDE_multiple
                end_index = (j + 1) * self.PDE_multiple
                combined_grid[start_index:end_index, i] = PDE_average[start_index:end_index, i] + (1 / self.h) * SSA_average[j, i]
        combined_grid[-1, :] = combined_grid[-2, :]

        print("Simulation completed")

        # Save simulation data including SSA events and PDE update times

        return SSA_average, PDE_average, combined_grid, all_SSA_events_logs, all_PDE_update_times

     
    def save_simulation_data(self, SSA_grid: np.ndarray, PDE_grid: np.ndarray, combined_grid: np.ndarray, all_SSA_events_logs: list, all_PDE_update_times: list, datadirectory='data'):
        if not os.path.exists(datadirectory):
            os.makedirs(datadirectory)
        params = {
            'domain_length': self.L,
            'compartment_number': self.SSA_M,
            'PDE_multiple': self.PDE_multiple,
            'total_time': self.total_time,
            'timestep': self.timestep,
            'threshold': self.threshold,
            'gamma': self.gamma,
            'deltax': self.deltax,
            'production_rate': self.production_rate,
            'degradation_rate': self.degradation_rate,
            'diffusion_rate': self.diffusion_rate,
            'threshold_conc': self.threshold_conc,
            'initial_SSA': self.SSA_initial.tolist(),
            'h': self.h,
        }
        np.savez(os.path.join(datadirectory, 'Hybrid_data'),
                SSA_grid=SSA_grid,
                PDE_grid=PDE_grid,
                combined_grid=combined_grid,
                time_vector=self.time_vector,
                SSA_X=self.SSA_X,
                PDE_X=self.PDE_X)
        
        # Save the lists of SSA events and PDE update times separately
        np.save(os.path.join(datadirectory, 'SSA_events_logs.npy'), np.array(all_SSA_events_logs, dtype=object))
        np.save(os.path.join(datadirectory, 'PDE_update_times.npy'), np.array(all_PDE_update_times, dtype=object))
        
        with open(os.path.join(datadirectory, "parameters.json"), 'w') as params_file:
            json.dump(params, params_file, indent=4)
        print("Data saved successfully")