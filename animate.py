import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import json
import seaborn as sns

def load_data(filename):
    Hybrid_data = np.load(filename + "Hybrid_data.npz")
    SSA_data = np.load(filename+"Pure_SSA_data.npz")

    parameters = json.load(open(filename + "parameters.json"))
    pure_SSA_parameters = json.load(open(filename + "parameters_pure_SSA.json"))

    return Hybrid_data, parameters, SSA_data, pure_SSA_parameters

def calculate_mass_continuous(data_grid, deltax):
    return np.sum(data_grid, axis=0) * deltax

def plot_animation(Hybrid_data, SSA_data, parameters):
    C_grid = Hybrid_data["PDE_grid"]
    D_grid = Hybrid_data["SSA_grid"]
    combined_grid = Hybrid_data["combined_grid"]
    SSA_X = Hybrid_data["SSA_X"]
    PDE_X = Hybrid_data["PDE_X"]
    time_vector = Hybrid_data["time_vector"]
    for key in SSA_data:
        print(key,"helloe")
    SSA_grid = SSA_data["SSA_grid"]

    for key in SSA_data:
        print(key, SSA_data[key].shape)
    h = parameters["h"]
    deltax = parameters["deltax"]
    production_rate = parameters["production_rate"]
    degradation_rate = parameters["degradation_rate"]
    concentration_threshold = parameters["threshold_conc"]
    domain_length = parameters["domain_length"]

    fig, ax = plt.subplots(figsize=(12, 8)) 

    # Bar plots for SSA
    bar_SSA = ax.bar(
        SSA_X, D_grid[:, 0] / h, width=h, color='blue', align='edge', label='Hybrid SSA (Bar Chart)', alpha=0.7
    )

    bar_pure_SSA = ax.bar(
        SSA_X, SSA_grid[:, 0] / h, width=h, color='cyan', align='edge', label='Pure SSA (Bar Chart)', alpha=0.5
    )

    # Line plots for PDE and combined
    line_PDE, = ax.plot(PDE_X, C_grid[:, 0], 'g--', label='Hybrid PDE', linewidth=2)
    line_combined, = ax.plot(PDE_X, combined_grid[:, 0], 'k--', label='Combined', linewidth=2)

    # Reference lines
    threshold_line = ax.axhline(y=concentration_threshold, color='purple', linestyle='--', label='Threshold', linewidth=1.5)
    steady_state_concentration = production_rate / degradation_rate
    steady_state_line = ax.axhline(
        y=steady_state_concentration,
        color='gray',
        linestyle='--',
        label='Steady State',
        linewidth=1.5,
    )

    # Axes and labels
    ax.set_xlabel('Spatial Domain', fontsize=12)
    ax.set_ylabel('Species Concentration', fontsize=12)
    ax.set_title('Hybrid Simulation', fontsize=14)
    ax.set_xlim(0, domain_length)
    y_max = max(np.max(combined_grid) * 1.1, steady_state_concentration * 1.1, concentration_threshold * 1.1)
    ax.set_ylim(-20, y_max)
    ax.grid(False)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def update(frame):
        for bar, height in zip(bar_SSA, D_grid[:, frame] / h):
            bar.set_height(height)
        for bar, height in zip(bar_pure_SSA, SSA_grid[:, frame] / h):
            bar.set_height(height)

        line_PDE.set_ydata(C_grid[:, frame])
        line_combined.set_ydata(combined_grid[:, frame])
        time_text.set_text(f'Time: {time_vector[frame]:.2f}')

        return (*bar_SSA, *bar_pure_SSA, line_combined, line_PDE, time_text, threshold_line, steady_state_line)

    ani = FuncAnimation(fig, update, frames=range(0, len(time_vector)), interval=40)

    # Legend and layout
    fig.subplots_adjust(right=0.8)
    ax.legend([
        bar_SSA[0], bar_pure_SSA[0], line_PDE, line_combined
    ], [
        "Hybrid SSA", "Pure SSA", "Hybrid PDE", "Hybrid SSA + PDE"
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

    Hybrid_data, parameters,  SSA_data, pure_SSA_parameters = load_data(filename)

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

    plot_animation(Hybrid_data, SSA_data, parameters)
    plot_total_mass(time_vector, combined_total_mass, domain_length, production_rate, degradation_rate, concentration_threshold)

if __name__ == '__main__':
    filename = 'simulation_data/'
    main(filename)