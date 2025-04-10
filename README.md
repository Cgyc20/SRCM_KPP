# 🧪 Hybrid SRCM Simulation of FKPP Dynamics

This project simulates the **FKPP (Fisher–Kolmogorov–Petrovsky–Piskunov)** reaction-diffusion equation using a **Hybrid Spatial Regime Conversion Method (SRCM)** designed and developed for my PhD project, this leans on the work of Kyanston and Yates in the Regime conversion method. This system dynamically switches between a **Stochastic Simulation Algorithm (SSA)** and a **deterministic PDE solver**, depending on local particle concentrations — ensuring efficiency **without sacrificing stochastic accuracy**.

This is designed in such a way that at the front of the wave we aim to apply the SSA while behind the wave we will apply the PDE approximation of the mean-field model.

---

## 📁 Project Structure

```
SRCM_KPP/
├── main.py                  # Run this to start the hybrid simulation
├── animate.py               # Run this to animate the results
├── src/
│   ├── SRCM.py              # Main hybrid model class with regime conversion logic
│   ├── Stochastic.py        # SSA and stochastic-only routines
│   └── c_class/             # C backend for fast propensity calculations
│       ├── C_functions.c
│       ├── C_functions.so   # Compiled shared object
│       ├── C_numerical_functions.c
│       ├── python_wrapper.py  # Python interface to C
├── simulation_data/         # Simulation output is saved here
├── tests/                   # Unit and integration tests
├── log.txt                  # Optional log output
├── Makefile                 # For compiling C backend
└── README.md                # This file
```

---

## 🚀 How to Run

### 1. ✅ **Run the simulation**

```bash
python main.py
```

This runs the hybrid SRCM simulation:
- FKPP dynamics are modelled using SSA in low-density regions and PDEs elsewhere.
- You can edit model parameters directly in `main.py` to adjust:
  - Diffusion rate
  - Reaction terms
  - Conversion thresholds (as a particle number per cell)
  - Simulation time and grid size


The simulation saves outputs to the `simulation_data/` directory and plots **total mass over time** to track conservation.

---

### 2. 🎥 **Animate the simulation**

After the simulation completes, run:

```bash
python animate.py
```

This will:
- Animate the hybrid and pure SSA solutions
- Plot SSA bars, PDE lines, and hybrid concentration profiles over time
- Overlay steady-state reference and threshold levels
- This uses the SSA as comparison, note the SSA is much slower to run. 

---

## ⚙️ Performance Optimisation

- The project uses **C code** for computational bottlenecks:
  - Propensity and stochastic reaction calculations
- You can compile the C backend using the included `Makefile`:
- This may be updated in the future to apply even more computationally expensive tasks. 

```bash
cd src/c_class
make
```

Make sure the resulting `C_functions.so` is present in the same folder as `python_wrapper.py`.

---

## 🧠 Scientific Background

The **Hybrid SRCM** method enables:
- Efficient simulation of spatial stochastic systems
- Regime switching based on particle thresholds. We have a transfer of mass between the continuous and discrete regimes. 
- Application to complex nonlinear models like **FKPP**, which exhibits traveling wave solutions

This project is suitable for testing hybrid modelling frameworks in theoretical biology, chemical kinetics, and spatial population dynamics. I am yet to develop code for generalised reaction-diffusion systems but this is a working progress.

---

## 🛠️ Dependencies

- Python 3.7+
- `numpy`
- `matplotlib`
- `ctypes` (standard library, used to interface with C)

You can install required packages via:

```bash
pip install numpy matplotlib scipy
```

---

## 📫 Contact

For questions, collaborations, or bug reports — feel free to open an issue or reach out directly. 