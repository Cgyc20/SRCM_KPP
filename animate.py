import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import seaborn as sns

class HybridSimulationPlotter:
    def __init__(self, filename):
        self.filename = filename
        self.load_data()
        self.extract_parameters()

    def load_data(self):
        self.hybrid_data = np.load(self.filename + "Hybrid_data.npz")
        self.ssa_data = np.load(self.filename + "Pure_SSA_data.npz")

        with open(self.filename + "parameters.json") as f:
            self.parameters = json.load(f)
        with open(self.filename + "parameters_pure_SSA.json") as f:
            self.pure_ssa_parameters = json.load(f)

    def extract_parameters(self):
        p = self.parameters
        self.h = p["h"]
        self.deltax = p["deltax"]
        self.production_rate = p["production_rate"]
        self.degradation_rate = p["degradation_rate"]
        self.threshold_conc = p["threshold_conc"]
        self.domain_length = p["domain_length"]

        self.time_vector = self.hybrid_data["time_vector"]
        self.C_grid = self.hybrid_data["PDE_grid"]
        self.D_grid = self.hybrid_data["SSA_grid"]
        self.combined_grid = self.hybrid_data["combined_grid"]
        self.SSA_X = self.hybrid_data["SSA_X"]
        self.PDE_X = self.hybrid_data["PDE_X"]
        self.SSA_grid = self.ssa_data["SSA_grid"]

    def calculate_total_mass(self):
        return np.sum(self.combined_grid, axis=0) * self.deltax

    def plot_animation(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Bar plots
        bar_hybrid_ssa = ax.bar(
            self.SSA_X, self.D_grid[:, 0] / self.h, width=self.h,
            color='blue', align='edge', label='Hybrid SSA', alpha=0.7
        )

        bar_pure_ssa = ax.bar(
            self.SSA_X, self.SSA_grid[:, 0] / self.h, width=self.h,
            color='cyan', align='edge', label='Pure SSA', alpha=0.5
        )

        # Line plots
        line_pde, = ax.plot(self.PDE_X, self.C_grid[:, 0], 'g--', label='Hybrid PDE', linewidth=2)
        line_combined, = ax.plot(self.PDE_X, self.combined_grid[:, 0], 'k--', label='Combined', linewidth=2)

        # Reference lines
        ax.axhline(y=self.threshold_conc, color='purple', linestyle='--', label='Threshold', linewidth=1.5)
        steady_state = self.production_rate / self.degradation_rate
        ax.axhline(y=steady_state, color='gray', linestyle='--', label='Steady State', linewidth=1.5)

        # Axes
        ax.set_xlabel('Spatial Domain')
        ax.set_ylabel('Species Concentration')
        ax.set_title('Hybrid Simulation')
        ax.set_xlim(0, self.domain_length)
        y_max = max(np.max(self.combined_grid), steady_state, self.threshold_conc) * 1.1
        ax.set_ylim(-20, y_max)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        def update(frame):
            for bar, height in zip(bar_hybrid_ssa, self.D_grid[:, frame] / self.h):
                bar.set_height(height)
            for bar, height in zip(bar_pure_ssa, self.SSA_grid[:, frame] / self.h):
                bar.set_height(height)
            line_pde.set_ydata(self.C_grid[:, frame])
            line_combined.set_ydata(self.combined_grid[:, frame])
            time_text.set_text(f'Time: {self.time_vector[frame]:.2f}')
            return (*bar_hybrid_ssa, *bar_pure_ssa, line_combined, line_pde, time_text)

        ani = FuncAnimation(fig, update, frames=len(self.time_vector), interval=40)

        # Legend
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)
        plt.show()

    def plot_total_mass(self):
        total_mass = self.calculate_total_mass()
        plt.figure(figsize=(8, 6))
        plt.plot(self.time_vector, total_mass, 'k--', label='Combined Total Mass', linewidth=2)
        plt.axhline(y=self.domain_length * (self.production_rate / self.degradation_rate),
                    color='gray', linestyle='--', label='Steady State', linewidth=1.5)
        plt.axhline(y=self.domain_length * self.threshold_conc,
                    color='purple', linestyle='--', label='Threshold', linewidth=1.5)
        plt.xlabel('Time')
        plt.ylabel('Total Mass')
        plt.title('Total Mass Over Time')
        plt.legend()
        plt.grid(False)
        plt.show()

    def run_all(self):
        sns.set_theme(style="whitegrid")
        self.plot_animation()
        self.plot_total_mass()


if __name__ == '__main__':
    plotter = HybridSimulationPlotter(filename='simulation_data/')
    plotter.run_all()