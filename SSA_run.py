import numpy as np
from src import SSA

def main():
    # Define parameters
    domain_length = 5
    compartment_length = 0.1
    PDE_multiple = 8
    total_time = 10
    timestep = 0.005
    particles_per_compartment_thresh = 50
    production_rate = 10
    degradation_rate = 0.01
    number_particles_per_cell = 10
    repeats = 200
    diffusion_rate = 1e-2

    # Derived parameters
    compartment_number = int(domain_length / compartment_length)
    deltax = compartment_length / PDE_multiple
    threshold_conc = particles_per_compartment_thresh / compartment_length
    h = compartment_length

    # Initial SSA values (all particles in the leftmost compartment)
    SSA_initial = np.zeros(compartment_number, dtype=int)
    SSA_initial[0] = number_particles_per_cell

    # Input dictionary (minimal for SSA)
    input_params = {
        'domain_length': domain_length,
        'compartment_number': compartment_number,
        'PDE_multiple': PDE_multiple,
        'total_time': total_time,
        'timestep': timestep,
        'threshold': particles_per_compartment_thresh,
        'gamma': None,  # not needed for pure SSA
        'degradation_rate': degradation_rate,
        'diffusion_rate': diffusion_rate,
        'h': h,
        'deltax': deltax,
        'production_rate': production_rate,
        'threshold_conc': threshold_conc,
        'SSA_initial': SSA_initial,
        'PDE_points': 0,
        'PDE_initial': None
    }

    # Create SSA model
    ssa_model = SSA(input_params)

    # Run a single SSA simulation (averaged over repeats)
    print("Running pure SSA simulation...")
    pure_SSA_average = ssa_model.run_simulation(repeats)

    # Save results
    ssa_model.save_simulation_data(
        filled_SSA_grid=pure_SSA_average,
        filename="SSA_data.npz",
        datadirectory="simulation_data"
    )

    print("Pure SSA simulation saved as simulation_data/Pure_SSA_data.npz")

if __name__ == "__main__":
    main()
