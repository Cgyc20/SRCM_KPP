# 🧪 Hybrid SRCM Simulation of FKPP Dynamics

This project simulates the **FKPP (Fisher–Kolmogorov–Petrovsky–Piskunov)** reaction-diffusion equation using a **Hybrid Spatial Regime Conversion Method (SRCM)**. It dynamically switches between a **Stochastic Simulation Algorithm (SSA)** and a **deterministic PDE solver**, depending on local particle concentrations — ensuring efficiency **without sacrificing stochastic fidelity**.

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
- FKPP dynamics are modeled using SSA in low-density regions and PDEs elsewhere.
- You can edit model parameters directly in `main.py` to adjust:
  - Diffusion rate
  - Reaction terms
  - Conversion thresholds
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

---

## ⚙️ Performance Optimisation

- The project uses **C code** for computational bottlenecks:
  - Propensity and stochastic reaction calculations
- You can compile the C backend using the included `Makefile`:

```bash
cd src/c_class
make
```

Make sure the resulting `C_functions.so` is present in the same folder as `python_wrapper.py`.

---

## 🧠 Scientific Background

The **Hybrid SRCM** method enables:
- Efficient simulation of spatial stochastic systems
- Regime switching based on particle thresholds (e.g., using the **RHS rule**)
- Application to complex nonlinear models like **FKPP**, which exhibits traveling wave solutions

This project is suitable for testing hybrid modeling frameworks in theoretical biology, chemical kinetics, and spatial population dynamics.

---

## 🛠️ Dependencies

- Python 3.7+
- `numpy`
- `matplotlib`
- `scipy`
- `ctypes` (standard library, used to interface with C)

You can install required packages via:

```bash
pip install numpy matplotlib scipy
```

---

## 📫 Contact

For questions, collaborations, or bug reports — feel free to open an issue or reach out directly.