import sys
import os
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

# Add the root dir to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from src
from src import PDE, SSA, Hybrid


def save_to_csv(folder, data):
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the file path
    file_path = os.path.join(folder, 'timing_results.csv')

    # Write the data to the CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Production Rate', 'Hybrid Time per Iteration (s)', 'SSA Time per Iteration (s)', 'PDE Time per Iteration (s)', 'Max Concentration'])
        writer.writerows(data)

    print(f"Results saved to {file_path}")


def plot_results(max_concentrations, hybrid_times_per_iter, ssa_times_per_iter, pde_times_per_iter, threshold_conc):
    plt.figure(figsize=(12, 8))

    # Plot Hybrid, SSA, and PDE times per iteration against max concentration
    plt.plot(max_concentrations, hybrid_times_per_iter, label='Hybrid Time per Iteration', marker='o', color='b')
    plt.plot(max_concentrations, ssa_times_per_iter, label='SSA Time per Iteration', marker='x', color='r')
    plt.plot(max_concentrations, pde_times_per_iter, label='PDE Time per Iteration', marker='s', color='g')

    # Add a vertical dotted line for the threshold concentration
    plt.axvline(x=threshold_conc, color='purple', linestyle='--', label='Threshold Concentration')

    # Labeling the axes
    plt.xlabel('Max Concentration', fontsize=14)
    plt.ylabel('Time per repeat (seconds)', fontsize=14)

    plt.ylim(0, max(np.max(ssa_times_per_iter), np.max(hybrid_times_per_iter), np.max(pde_times_per_iter)) + 0.1)
    plt.xlim(0, max(max_concentrations) + 50)

    # Add a legend with a larger font size
    plt.legend(fontsize=12)

    # Display the plot
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    # Define the input parameters for the models
    domain_length = 5
    compartment_length = 0.1
    PDE_multiple = 8
    total_time = 10
    timestep = 0.008
    particles_per_compartment_thresh = 20
    gamma = 1
    degradation_rate = 0.01
    number_particles_per_cell = 5
    repeats = 10
    diffusion_rate = 1e-2

    # Derived parameters
    compartment_number = int(domain_length / compartment_length)
    deltax = compartment_length / PDE_multiple
    threshold_conc = particles_per_compartment_thresh / compartment_length
    h = compartment_length

    # Initial SSA values
    SSA_initial = np.zeros(compartment_number, dtype=int)
    SSA_initial[0] = number_particles_per_cell  # All particles in the first compartment

    # Initial PDE values
    PDE_points = compartment_number * PDE_multiple
    PDE_initial = np.zeros(PDE_points)  # Initialize PDE grid with zeros
    PDE_initial[:PDE_multiple] = number_particles_per_cell/h  # Set the first PDE compartment with the initial number of particles

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
        'production_rate': 1,  # Initial production rate (this will be updated in the loop)
        'threshold_conc': threshold_conc,
        'SSA_initial': SSA_initial,
        'PDE_points': PDE_points,
        'PDE_initial': PDE_initial
    }

    # Lists to store times and max concentrations
    hybrid_times_per_iter = []
    ssa_times_per_iter = []
    pde_times_per_iter = []
    max_concentrations = []
    production_rates = []

    # Folder to save the results
    result_folder = 'timing_results'

    # Loop through production rates from 0.5 to 10
    for production_rate in np.arange(0.5, 10.5, 0.5):
        # Update the production rate in the input parameters
        input_params['production_rate'] = production_rate

        # Calculate the max concentration as production_rate / degradation_rate
        max_concentration = production_rate / degradation_rate
        max_concentrations.append(max_concentration)
        production_rates.append(production_rate)

        # Create instances of the models
        hybrid_model = Hybrid(input_params)
        ssa_model = SSA(input_params)

        # Measure time for the Hybrid model
        start_time = time.perf_counter()
        hybrid_model.run_simulation(repeats)
        hybrid_time = time.perf_counter() - start_time

        # Measure time for the SSA model
        start_time = time.perf_counter()
        ssa_model.run_simulation(repeats)
        ssa_time = time.perf_counter() - start_time

        # Measure time for the PDE model (run multiple times for fair comparison)
        pde_times = []
        PDE_repeats = 5
        pde_model = PDE(input_params)  # Create new instance each time
        for _ in range(PDE_repeats):
            
            start_time = time.perf_counter()
            pde_model.run_simulation()
            pde_time_single = time.perf_counter() - start_time
            pde_times.append(pde_time_single)
        
        pde_time_total = sum(pde_times)

        # Calculate time per iteration for all models
        hybrid_time_per_iter = hybrid_time / repeats
        ssa_time_per_iter = ssa_time / repeats
        pde_time_per_iter = pde_time_total / PDE_repeats

        # Append the results to the lists
        hybrid_times_per_iter.append(hybrid_time_per_iter)
        ssa_times_per_iter.append(ssa_time_per_iter)
        pde_times_per_iter.append(pde_time_per_iter)

        # Print the timing for this production rate
        print(f"Production rate {production_rate}: Max Concentration = {max_concentration:.2f}, "
              f"Hybrid time per iteration = {hybrid_time_per_iter:.4f}s, "
              f"SSA time per iteration = {ssa_time_per_iter:.4f}s, "
              f"PDE time per iteration = {pde_time_per_iter:.4f}s")

    # Save the results to a CSV file
    results = list(zip(production_rates, hybrid_times_per_iter, ssa_times_per_iter, pde_times_per_iter, max_concentrations))
    save_to_csv(result_folder, results)

    # Plot the results
    plot_results(max_concentrations, hybrid_times_per_iter, ssa_times_per_iter, pde_times_per_iter, threshold_conc)

    # Print final summary
    print("\nFinal timing results:")
    for i, prod_rate in enumerate(production_rates):
        print(f"Production rate {prod_rate}: Max Concentration = {max_concentrations[i]:.2f}, "
              f"Hybrid time per iteration = {hybrid_times_per_iter[i]:.4f}s, "
              f"SSA time per iteration = {ssa_times_per_iter[i]:.4f}s, "
              f"PDE time per iteration = {pde_times_per_iter[i]:.4f}s")

if __name__ == "__main__":
    main()