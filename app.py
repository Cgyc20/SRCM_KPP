import tkinter as tk
from tkinter import ttk
import numpy as np
from src import Hybrid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import json
import os

class HybridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Simulation GUI")

        # Default params
        self.params = {
            'domain_length': 5,
            'compartment_length': 0.1,
            'PDE_multiple': 8,
            'total_time': 10,
            'timestep': 0.008,
            'particles_per_compartment_thresh': 50,
            'gamma': 5,
            'production_rate': 10.0,
            'degradation_rate': 0.01,
            'number_particles_per_cell': 1,
            'repeats': 50,
            'diffusion_rate': 0.01,
        }

        self.entries = {}  # Store widgets
        self.build_gui()

    def build_gui(self):
        row = 0
        for key, val in self.params.items():
            tk.Label(self.root, text=key).grid(row=row, column=0, sticky="w", padx=5, pady=3)
            if isinstance(val, float):
                if key == 'compartment_length':
                    max_val = self.params['domain_length'] / 2
                    scale = tk.Scale(self.root, from_=0.05, to=max_val, resolution=0.05,
                                    orient=tk.HORIZONTAL, length=200)
                    scale.set(val)
                    scale.grid(row=row, column=1)
                    self.entries[key] = scale
                else:
                    scale = tk.Scale(self.root, from_=0.0, to=20.0, resolution=0.001,
                                    orient=tk.HORIZONTAL, length=200)
                    scale.set(val)
                    scale.grid(row=row, column=1)
                    self.entries[key] = scale
                else:
                    entry = ttk.Entry(self.root)
                    entry.insert(0, str(val))
                    entry.grid(row=row, column=1)
                    self.entries[key] = entry
            row += 1

        ttk.Button(self.root, text="Run Simulation", command=self.run_simulation).grid(row=row, column=0, pady=10)
        ttk.Button(self.root, text="Visualise", command=self.plot_results).grid(row=row, column=1, pady=10)

    def get_params(self):
        parsed = {}
        for key, widget in self.entries.items():
            if isinstance(widget, tk.Scale):
                parsed[key] = float(widget.get())
            else:
                val = widget.get()
                parsed[key] = float(val) if '.' in val else int(val)
        return parsed

    def run_simulation(self):
        input_params = self.get_params()

        compartment_number = int(input_params['domain_length'] / input_params['compartment_length'])
        deltax = input_params['compartment_length'] / input_params['PDE_multiple']
        threshold_conc = input_params['particles_per_compartment_thresh'] / input_params['compartment_length']
        h = input_params['compartment_length']

        SSA_initial = np.zeros(compartment_number, dtype=int)
        SSA_initial[0] = compartment_number * input_params['number_particles_per_cell']

        full_input = {
            'domain_length': input_params['domain_length'],
            'compartment_number': compartment_number,
            'PDE_multiple': input_params['PDE_multiple'],
            'total_time': input_params['total_time'],
            'timestep': input_params['timestep'],
            'threshold': input_params['particles_per_compartment_thresh'],
            'gamma': input_params['gamma'],
            'degradation_rate': input_params['degradation_rate'],
            'diffusion_rate': input_params['diffusion_rate'],
            'h': h,
            'deltax': deltax,
            'production_rate': input_params['production_rate'],
            'threshold_conc': threshold_conc,
            'SSA_initial': SSA_initial
        }

        hybrid_model = Hybrid(full_input)
        SSA_avg, PDE_avg, combined = hybrid_model.run_simulation(input_params['repeats'])
        hybrid_model.save_simulation_data(SSA_avg, PDE_avg, combined, 'simulation_data')

        tk.messagebox.showinfo("Simulation Done", "Simulation completed and data saved!")

    def plot_results(self):
        filename = 'simulation_data/'
        data = np.load(filename + "Hybrid_data.npz")
        params = json.load(open(filename + "parameters.json"))

        C_grid = data["PDE_grid"]
        combined_grid = data["combined_grid"]
        PDE_X = data["PDE_X"]
        time_vector = data["time_vector"]
        deltax = params["deltax"]
        domain_length = params["domain_length"]
        prod_rate = params["production_rate"]
        deg_rate = params["degradation_rate"]
        threshold = params["threshold_conc"]

        def update(frame):
            line1.set_ydata(C_grid[:, frame])
            line2.set_ydata(combined_grid[:, frame])
            time_text.set_text(f'Time: {time_vector[frame]:.2f}')
            return line1, line2, time_text

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
        line1, = ax.plot(PDE_X, C_grid[:, 0], 'g--', label='PDE')
        line2, = ax.plot(PDE_X, combined_grid[:, 0], 'k-', label='Combined')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        ax.axhline(y=threshold, color='purple', linestyle='--', label='Threshold')
        ax.axhline(y=prod_rate / deg_rate, color='gray', linestyle='--', label='Steady State')
        ax.set_ylim(-20, np.max(combined_grid) * 1.2)
        ax.set_xlim(0, domain_length)
        ax.set_title("Hybrid Simulation")
        ax.legend()

        ani = FuncAnimation(fig, update, frames=len(time_vector), interval=50)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = HybridApp(root)
    root.mainloop()