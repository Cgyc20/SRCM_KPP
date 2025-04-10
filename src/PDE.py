import numpy as np
from tqdm import tqdm
import os
import json

class PDE:
    def __init__(self, input):
        # Initialise the parameters from input dictionary
        self.L = input['domain_length']
        self.PDE_points = input['PDE_points']
        self.total_time = input['total_time']
        self.timestep = input['timestep']
        self.production_rate = input['production_rate']
        self.degradation_rate = input['degradation_rate']
        self.diffusion_rate = input['diffusion_rate']
        self.PDE_initial_conditions = input['PDE_initial']

        self.deltax = self.L / self.PDE_points
        self.PDE_X = np.linspace(0, self.L, self.PDE_points)
        self.steady_state = self.production_rate / self.degradation_rate
        self.DX_NEW = self.create_finite_difference()
        self.time_vector = np.arange(0, self.total_time, self.timestep)

        self.PDE_grid = np.zeros((self.PDE_points, len(self.time_vector)))
        self.PDE_grid[:, 0] = self.PDE_initial_conditions

        print("Successfully initialized the PDE model")

    def create_finite_difference(self):
        self.DX = np.zeros((self.PDE_points, self.PDE_points), dtype=int)
        self.DX[0, 0], self.DX[-1, -1] = -1, -1
        self.DX[0, 1], self.DX[-1, -2] = 1, 1
        for i in range(1, self.DX.shape[0] - 1):
            self.DX[i, i] = -2
            self.DX[i, (i + 1)] = 1
            self.DX[i, (i - 1)] = 1
        return self.DX

    def RHS_derivative(self, old_vector):
        """The RHS, i.e., du/dt approximation"""
        dudt = np.zeros_like(old_vector)
        nabla = self.DX_NEW
        dudt = (
            self.diffusion_rate * (1 / self.deltax) ** 2 * nabla @ old_vector
            + self.production_rate * old_vector
            - self.degradation_rate * old_vector ** 2
        )
        return dudt

    def RK4(self, old_vector):
        """The Runge-Kutta 4th order method"""
        k1 = self.RHS_derivative(old_vector)
        k2 = self.RHS_derivative(old_vector + 0.5 * self.timestep * k1)
        k3 = self.RHS_derivative(old_vector + 0.5 * self.timestep * k2)
        k4 = self.RHS_derivative(old_vector + self.timestep * k3)
        return old_vector + self.timestep * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def run_simulation(self):
        for i in tqdm(range(len(self.time_vector) - 1), desc="Running PDE simulation"):
            self.PDE_grid[:, i + 1] = self.RK4(self.PDE_grid[:, i])
        print("Simulation completed")
        return self.PDE_grid

    def save_simulation_data(self, PDE_grid, filename = "PDE_data.npz",datadirectory='data'):
        if not os.path.exists(datadirectory):
            os.makedirs(datadirectory)
        params = {
            'domain_length': self.L,
            'PDE_points': self.PDE_points,
            'total_time': self.total_time,
            'timestep': self.timestep,
            'production_rate': self.production_rate,
            'degradation_rate': self.degradation_rate,
            'diffusion_rate': self.diffusion_rate,
        }
        np.savez(
            os.path.join(datadirectory, filename),
            PDE_grid=PDE_grid,
            PDE_X=self.PDE_X,
            time_vector=self.time_vector,
            parameters=params,
        )
        print("Data saved successfully as ", filename)