import numpy as np
import time
import csv
from src import Hybrid

def main():
    # Define fixed input parameters
    domain_length = 5
    compartment_length = 0.1
    PDE_multiple = 8
    total_time = 10
    timestep = 0.005
    particles_per_compartment_thresh = 50
    production_rate = 10
    degradation_rate = 0.01
    number_particles_per_cell = 10
    repeats = 50
    diffusion_rate = 1e-2

    # Derived parameters
    compartment_number = int(domain_length / compartment_length)
    deltax = compartment_length / PDE_multiple
    threshold_conc = particles_per_compartment_thresh / compartment_length
    h = compartment_length

    # Initial SSA values
    SSA_initial = np.zeros(compartment_number, dtype=int)
    SSA_initial[0] = number_particles_per_cell

    # Initial PDE values
    PDE_points = compartment_number * PDE_multiple
    PDE_initial = np.zeros(PDE_points)
    PDE_initial[:PDE_multiple] = number_particles_per_cell / h

    # Loop over gamma values
    gamma_values = np.arange(0.5, 10.5, 0.5)  # 0.5, 1.0, ..., 10.0

    timings = []  # store timings for each gamma

    for gamma in gamma_values:
        print(f"Running Hybrid model with gamma = {gamma}")

        # Input dictionary
        input_params = {
            'domain_length': domain_length,
            'compartment_number': compartment_number,
            'PDE_multiple': PDE_multiple,
            'total_time': total_time,
            'timestep': timestep,
            'threshold': particles_per_compartment_thresh,
            'gamma': gamma,
            'degradation_rate': degradation_rate,
            'diffusion_rate': diffusion_rate,
            'h': h,
            'deltax': deltax,
            'production_rate': production_rate,
            'threshold_conc': threshold_conc,
            'SSA_initial': SSA_initial,
            'PDE_points': PDE_points,
            'PDE_initial': PDE_initial
        }

        # Create Hybrid model
        hybrid_model = Hybrid(input_params, use_stochastic_init=False)

        # Time the simulation
        start_time = time.time()
        SSA_average, PDE_average, combined_grid = hybrid_model.run_simulation(repeats)
        end_time = time.time()
        elapsed_time = end_time - start_time

        timings.append((gamma, elapsed_time))
        print(f"Gamma={gamma} finished in {elapsed_time:.2f} seconds")

        # Save results for this gamma
        hybrid_model.save_simulation_data(
            SSA_grid=SSA_average,
            PDE_grid=PDE_average,
            combined_grid=combined_grid,
            filename=f"Hybrid_data_gamma_{gamma}.npz",
            datadirectory="simulation_data"
        )

    # Save timings as CSV
    with open("simulation_data/Hybrid_timings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma", "time_seconds"])
        writer.writerows(timings)

    # Also save timings as numpy for easier loading in Python
    np.savez("simulation_data/Hybrid_timings.npz", gamma=gamma_values, timings=[t for _, t in timings])


if __name__ == "__main__":
    main()
