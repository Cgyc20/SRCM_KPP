import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import json
import seaborn as sns

def load_data(filename):
    Hybrid_data = np.load(filename + "Hybrid_data.npz")
    parameters = json.load(open(filename + "parameters.json"))
    return Hybrid_data, parameters

def calculate_mass_continuous(data_grid, deltax):
    return np.sum(data_grid, axis=0) * deltax

def plot_animation(Hybrid_data, parameters):
    C_grid = Hybrid_data["PDE_grid"]
    combined_grid = Hybrid_data["combined_grid"]
    PDE_X = Hybrid_data["PDE_X"]
    time_vector = Hybrid_data["time_vector"]

    h = parameters["h"]
    deltax = parameters["deltax"]
    production_rate = parameters["production_rate"]
    degradation_rate = parameters["degradation_rate"]
    concentration_threshold = parameters["threshold_conc"]
    domain_length = parameters["domain_length"]

    fig, ax = plt.subplots(figsize=(12, 8)) 

    line_PDE, = ax.plot(PDE_X, C_grid[:, 0], 'g--', label='Hybrid PDE', linewidth=2)
    line_combined, = ax.plot(PDE_X, combined_grid[:, 0], 'k--', label='Combined', linewidth=2)
    threshold_line = ax.axhline(y=concentration_threshold, color='purple', linestyle='--', label='Threshold', linewidth=1.5)

    ax.set_xlabel('Spatial Domain', fontsize=12)
    ax.set_ylabel('Species Concentration', fontsize=12)
    ax.set_title('Hybrid Simulation', fontsize=14)
    ax.set_xlim(0, domain_length)
    ax.set_ylim(0, max(np.max(combined_grid) * 1.1, concentration_threshold * 1.1))
    ax.grid(False)  # Disable grid

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    steady_state_concentration = production_rate / degradation_rate
    y_max = max(np.max(combined_grid) * 1.1, steady_state_concentration * 1.1, concentration_threshold * 1.1)
    ax.set_ylim(-20, y_max)

    steady_state_line = ax.axhline(
        y=steady_state_concentration,
        color='gray',
        linestyle='--',
        label='Steady State',
        linewidth=1.5,
    )

    def update(frame):
        line_PDE.set_ydata(C_grid[:, frame])  # Hybrid PDE
        line_combined.set_ydata(combined_grid[:, frame])  # Hybrid SSA + PDE
        time_text.set_text(f'Time: {time_vector[frame]:.2f}')
        return line_combined, line_PDE, time_text, threshold_line, steady_state_line

    ani = FuncAnimation(fig, update, frames=range(0, len(time_vector), 1), interval=40)

    fig.subplots_adjust(right=0.8)
    ax.legend([
        line_PDE, line_combined
    ], [
        "Hybrid PDE", "Hybrid SSA + PDE"
    ], loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

    plt.show()

def plot_total_mass(time_vector, combined_total_mass, domain_length, production_rate, degradation_rate, concentration_threshold):
    plt.figure(figsize=(8, 6))

    plt.plot(time_vector, combined_total_mass, 'k--', label='Combined (Dashed)', linewidth=2)
    plt.axhline(y=domain_length * (production_rate / degradation_rate), color='gray', linestyle='--', label='Steady State', linewidth=1.5)
    plt.axhline(y=domain_length * concentration_threshold, color='purple', linestyle='--', label='Threshold', linewidth=1.5)

    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Total Mass', fontsize=12)
    plt.title('Total Mass over Time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(False)  # Disable grid
    plt.show()

def main(filename):
    sns.set_theme(style="whitegrid")

    Hybrid_data, parameters = load_data(filename)

    C_grid = Hybrid_data["PDE_grid"]
    combined_grid = Hybrid_data["combined_grid"]
    PDE_X = Hybrid_data["PDE_X"]
    time_vector = Hybrid_data["time_vector"]

    h = parameters["h"]
    deltax = parameters["deltax"]
    production_rate = parameters["production_rate"]
    degradation_rate = parameters["degradation_rate"]
    concentration_threshold = parameters["threshold_conc"]
    domain_length = parameters["domain_length"]

    combined_total_mass = calculate_mass_continuous(combined_grid, deltax)

    plot_animation(Hybrid_data, parameters)
    plot_total_mass(time_vector, combined_total_mass, domain_length, production_rate, degradation_rate, concentration_threshold)

if __name__ == '__main__':
    filename = 'simulation_data/'
    main(filename)