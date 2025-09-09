import matplotlib.pyplot as plt

# Max concentrations
max_concentrations = [
    50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0,
    550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0
]

# Hybrid times per iteration (seconds)
hybrid_times = [
    0.2285, 0.2958, 0.5188, 1.2690, 0.5228, 0.3849, 0.3442, 0.3171, 0.2986, 0.2924,
    0.3015, 0.2990, 0.2838, 0.2922, 0.2919, 0.2856, 0.3004, 0.2830, 0.2774, 0.2772
]

# SSA times per iteration (seconds)
ssa_times = [
    0.0099, 0.0359, 0.0875, 0.1544, 0.2820, 0.4356, 0.6134, 0.8896, 1.1285, 1.5554,
    1.8719, 2.4158, 3.0329, 3.6039, 4.5698, 5.2942, 6.0518, 6.6525, 7.8043, 8.6024
]

# PDE times (seconds)
pde_times = [
    0.49238687503384426, 0.4793872499722056, 0.4722549170255661, 0.47232437500497326,
    0.4735122909769416, 0.4782889580237679, 0.487584209011402, 0.4734212920302525,
    0.47291808301815763, 0.47243720799451694, 0.4739181659533642, 0.4738002910162322,
    0.4731027919915505, 0.5045776249608025, 0.4735218749847263, 0.4736756670172326,
    0.4816228750278242, 0.4712347919703461, 0.4843850829638541, 0.473698791989591
]

# Threshold concentration (from your simulations)
threshold_conc = 200.0

# Plot all three timing curves
plt.figure(figsize=(10,6))
plt.plot(max_concentrations, hybrid_times, label='Hybrid Time per Iteration', marker='o', color='b')
plt.plot(max_concentrations, ssa_times, label='SSA Time per Iteration', marker='x', color='r')
plt.plot(max_concentrations, pde_times, label='PDE Time', marker='s', color='m')
plt.axvline(x=threshold_conc, color='g', linestyle='--', label='Threshold Concentration')

plt.xlabel('Max Concentration')
plt.ylabel('Time per Iteration (s)')
plt.title('Simulation Timing Comparison: Hybrid vs SSA vs PDE')
plt.legend(fontsize=12)
plt.grid(False)
plt.xlim(0, max(max_concentrations)*1.05)


plt.show()
