import numpy as np
from src import Naive_Hybrid, PDE, SSA, Hybrid

def main():
    # Define the input parameters for the Hybrid model
    domain_length = 5
    compartment_length = 0.1
    PDE_multiple = 8
    total_time = 10
    timestep = 0.008
    particles_per_compartment_thresh = 25
    gamma = 5
    production_rate = 10
    degradation_rate = 0.01
    number_particles_per_cell = 100
    repeats = 100
    diffusion_rate = 1e-2

    # Derived parameters
    compartment_number = int(domain_length / compartment_length)
    deltax = compartment_length / PDE_multiple
    threshold_conc = particles_per_compartment_thresh / compartment_length
    h = compartment_length

    # Initial SSA values
    # Initialize all mass in the leftmost compartment
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
        'production_rate': production_rate,
        'threshold_conc': threshold_conc,
        'SSA_initial': SSA_initial,
        'PDE_points': PDE_points,
        'PDE_initial': PDE_initial
    }

    # Create an instance of the Hybrid class
    hybrid_model = Hybrid(input_params)

    # Create two instances of the SSA class
    SSA_model = SSA(input_params)


    # Create an instance of the PDE class
    PDE_model = PDE(input_params)

    Naive_hybrid_model = Naive_Hybrid(input_params)

    # Run the simulations
    SSA_average, PDE_average, combined_grid = hybrid_model.run_simulation(repeats)
   
    # SSA_average_naive, PDE_average_naive, combined_grid_naive = Naive_hybrid_model.run_simulation(repeats)
    pure_SSA_average = SSA_model.run_simulation(repeats)

    PDE_results = PDE_model.run_simulation()

    # Save the simulation data
    hybrid_model.save_simulation_data(
        SSA_grid=SSA_average,
        PDE_grid=PDE_average,
        combined_grid=combined_grid,
        filename="Hybrid_data.npz",
        datadirectory='simulation_data'
    )

    # Save the first SSA simulation
    SSA_model.save_simulation_data(
        filled_SSA_grid=pure_SSA_average,
        filename="Pure_SSA_data.npz",
        datadirectory='simulation_data',
    )


    # Save the PDE simulation
    PDE_model.save_simulation_data(
        PDE_grid=PDE_results,
        filename="PDE_data.npz",
        datadirectory='simulation_data',
    )


if __name__ == "__main__":
    main()