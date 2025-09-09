import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_results(max_concentrations, hybrid_times_per_iter, ssa_times_per_iter, threshold_conc):
    plt.figure(figsize=(10, 6))

    # Plot Hybrid and SSA times per iteration against max concentration
    plt.plot(max_concentrations, hybrid_times_per_iter, label='Hybrid Time per Iteration', marker='o', color='b')
    plt.plot(max_concentrations, ssa_times_per_iter, label='SSA Time per Iteration', marker='x', color='r')

    # Add a vertical dotted line for the threshold concentration
    plt.axvline(x=threshold_conc, color='g', linestyle='--', label='Threshold Concentration')

    # Labeling the axes
    plt.xlabel('Max Concentration', fontsize=14)  # Increase font size of labels
    plt.ylabel('Time per repeat (seconds)', fontsize=14)  # Increase font size of labels

    plt.ylim(0,np.max(ssa_times_per_iter)+0.1)
    plt.xlim(0,1010)
    # Add a legend with a larger font size
    plt.legend(fontsize=12)  # Increase font size of legend

    # Display the plot
    plt.grid(False)
    plt.show()

if __name__ == '__main__':


    production_rate = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    max_concentrations = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450])
    Hybrid_iter = np.array([0.406, 0.4119, 0.652, 2.84, 0.899, 0.576, 0.4734, 0.501, 0.44002])
    SSA_iter = np.array([0.0146, 0.044, 0.0951, 0.2091, 0.301, 0.4447, 0.6711, 0.9661, 1.2186])

    threshold_conc = 200


    plot_results(max_concentrations, Hybrid_iter, SSA_iter,threshold_conc)