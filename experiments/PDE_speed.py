import sys
import os
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

# Add the root dir to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the PDE class
from src import PDE


def save_to_csv(folder, data):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, 'pde_timing_results.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Production Rate', 'PDE Time (s)', 'Max Concentration'])
        writer.writerows(data)
    print(f"Results saved to {file_path}")


def main():
    # Model parameters
    domain_length = 5
    compartment_length = 0.1
    PDE_multiple = 8
    total_time = 10
    timestep = 0.008
    particles_per_compartment_thresh = 20
    number_particles_per_cell = 5
    repeats = 5  # only one run per production rate for timing
    diffusion_rate = 1e-2
    degradation_rate = 0.01

    # Derived parameters
    compartment_number = int(domain_length / compartment_length)
    deltax = compartment_length / PDE_multiple
    h = compartment_length
    threshold_conc = particles_per_compartment_thresh / compartment_length

    # Initial PDE values
    PDE_points = compartment_number * PDE_multiple
    PDE_initial = np.zeros(PDE_points)
    PDE_initial[:PDE_multiple] = number_particles_per_cell / h

    input_params = {
        'domain_length': domain_length,
        'compartment_number': compartment_number,
        'PDE_multiple': PDE_multiple,
        'total_time': total_time,
        'timestep': timestep,
        'diffusion_rate': diffusion_rate,
        'degradation_rate': degradation_rate,
        'h': h,
        'deltax': deltax,
        'production_rate': 1,  # placeholder, updated in loop
        'threshold_conc': threshold_conc,
        'PDE_points': PDE_points,
        'PDE_initial': PDE_initial
    }

    pde_times = []
    max_concentrations = []

    # Folder to save results
    result_folder = 'pde_timing_results'

    for production_rate in np.arange(0.5, 10.5, 0.5):
        input_params['production_rate'] = production_rate
        max_concentration = production_rate / degradation_rate
        max_concentrations.append(max_concentration)

        PDE_model = PDE(input_params)

        start_time = time.perf_counter()
        PDE_model.run_simulation()
        elapsed_time = time.perf_counter() - start_time

        pde_times.append(elapsed_time)

        print(f"Production rate {production_rate:.1f}: Max Concentration = {max_concentration:.2f}, PDE time = {elapsed_time:.4f}s")

    # Save results to CSV
    results = list(zip(np.arange(0.5, 10.5, 0.5), pde_times, max_concentrations))
    save_to_csv(result_folder, results)

if __name__ == "__main__":
    main()
