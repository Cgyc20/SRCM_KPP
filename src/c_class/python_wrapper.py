import numpy as np
import ctypes

class CFunctionWrapper:
    def __init__(self, params, library_path="C_functions.so"):
        """
        Initialize the wrapper by loading the C library and setting parameters.

        Parameters:
            params (dict): Dictionary containing the following keys:
                - SSA_M (int): Number of SSA compartments.
                - PDE_multiple (int): Number of PDE points per SSA compartment.
                - deltax (float): Spatial discretization step size.
                - h (float): Grid size or thresholding parameter.
                - threshold (float): Minimum mass threshold for significance.
                - production_rate (float): Rate of production.
                - degradation_rate_h (float): Degradation rate scaled by h.
                - jump_rate (float): Rate of stochastic jumps.
                - gamma (float): Nonlinear scaling or interaction parameter.
            library_path (str): Path to the compiled C library.
        """
        self.SSA_M = params['SSA_M']
        self.PDE_multiple = params['PDE_multiple']
        self.deltax = params['deltax']
        self.h = params['h']
        self.threshold = params['threshold']
        self.production_rate = params['production_rate']
        self.degradation_rate_h = params['degradation_rate_h']
        self.jump_rate = params['jump_rate']
        self.gamma = params['gamma']

        self.lib = ctypes.CDLL(library_path)

        # Define argument types for the C functions
        self.lib.ApproximateMassLeftHand.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_float
        ]
        self.lib.BooleanMass.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_float
        ]
        self.lib.BooleanThresholdMass.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
            ctypes.c_float, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.c_float
        ]
        self.lib.FineGridSSAMass.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.CalculatePropensity.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int), ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]

        

        print(f"Loaded C library from {library_path}")

    def __str__(self):
        return f"CFunctionWrapper(library_path={self.lib._name}) for the Hybrid method"

    def approximate_mass_left_hand(self, PDE_list):
        """
        Computes the approximate PDE mass at each SSA compartment using the left-hand rule.

        Parameters:
            PDE_list (array-like): Values of the PDE grid.

        Returns:
            np.ndarray: Approximate PDE mass in each SSA compartment.
        """
        approximate_PDE_mass = np.zeros(self.SSA_M, dtype=np.float32)
        PDE_list = np.array(PDE_list, dtype=np.float32)

        self.lib.ApproximateMassLeftHand(
            ctypes.c_int(self.SSA_M),
            ctypes.c_int(self.PDE_multiple),
            PDE_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            approximate_PDE_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(self.deltax)
        )

        return approximate_PDE_mass

    def calculate_total_mass(self, PDE_list, SSA_list):
        """
        Computes the total combined mass of PDE and SSA at each SSA compartment.

        Parameters:
            PDE_list (array-like): PDE values across the domain.
            SSA_list (array-like): SSA values per compartment.

        Returns:
            tuple:
                - combined_list (np.ndarray): Total mass per compartment (PDE + SSA).
                - approximate_PDE_mass (np.ndarray): The approximate PDE mass component.
        """
        PDE_list = np.array(PDE_list, dtype=np.float32)
        SSA_list = np.array(SSA_list, dtype=np.int32)

        approximate_PDE_mass = self.approximate_mass_left_hand(PDE_list)
        combined_list = np.add(SSA_list, approximate_PDE_mass)

        combined_list = np.array(combined_list, dtype=np.float32)
        return combined_list, approximate_PDE_mass

    def boolean_low_limit(self, PDE_list):
        """
        Computes boolean masks for significant PDE and SSA mass based on the minimum mass condition.

        Parameters:
            PDE_list (array-like): PDE concentration values.

        Returns:
            tuple:
                - min_mass_PDE_bool_list (np.ndarray): Mask for PDE domain (0 if mass < 1/h, else 1).
                - min_mass_compartment_bool_list (np.ndarray): Mask for SSA compartments (0 if any PDE bool is 0).
        """
        PDE_list = np.array(PDE_list, dtype=np.float32)
        PDE_length = self.SSA_M * self.PDE_multiple
        
        assert len(PDE_list) == PDE_length, "PDE_list length mismatch."

        min_mass_PDE_bool_list = np.zeros(PDE_length, dtype=np.int32)
        min_mass_compartment_bool_list = np.zeros(self.SSA_M, dtype=np.int32)

        self.lib.BooleanMass(
            ctypes.c_int(self.SSA_M),
            ctypes.c_int(PDE_length),
            ctypes.c_int(self.PDE_multiple),
            PDE_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            min_mass_PDE_bool_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            min_mass_compartment_bool_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_float(self.h)
        )

        return min_mass_PDE_bool_list, min_mass_compartment_bool_list

    def boolean_threshold_mass(self, combined_list):
        """
        Computes boolean masks based on a threshold for total mass.

        Parameters:
            combined_list (array-like): Combined SSA and PDE mass values.

        Returns:
            tuple:
                - compartment_bool_list (np.ndarray): Mask for SSA compartments (1 if mass > threshold).
                - PDE_bool_list (np.ndarray): Mask for PDE domain (1 if mass > threshold).
        """
        combined_list = np.array(combined_list, dtype=np.float32)
        compartment_bool_list = np.zeros(self.SSA_M, dtype=np.int32)
        PDE_length = self.SSA_M * self.PDE_multiple
        PDE_bool_list = np.zeros(PDE_length, dtype=np.int32)

        self.lib.BooleanThresholdMass(
            ctypes.c_int(self.SSA_M),
            ctypes.c_int(PDE_length),
            ctypes.c_int(self.PDE_multiple),
            combined_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(self.h),
            compartment_bool_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            PDE_bool_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_float(self.threshold)
        )

        return compartment_bool_list, PDE_bool_list

    def fine_grid_ssa_mass(self, SSA_mass):
        """
        Spreads SSA mass from coarse compartments into a fine PDE grid representation.

        Parameters:
            SSA_mass (array-like): SSA values per compartment.

        Returns:
            np.ndarray: Fine-grained representation of SSA mass on PDE grid.
        """
        SSA_mass = np.array(SSA_mass, dtype=np.int32)
        PDE_grid_length = self.SSA_M * self.PDE_multiple
        fine_SSA_Mass = np.zeros(PDE_grid_length, dtype=np.float32)

        self.lib.FineGridSSAMass(
            SSA_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(PDE_grid_length),
            ctypes.c_int(self.SSA_M),
            ctypes.c_int(self.PDE_multiple),
            ctypes.c_float(self.h),
            fine_SSA_Mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        return fine_SSA_Mass

    def calculate_propensity(self, PDE_list, SSA_list):
        """
        Computes the reaction propensities in each SSA compartment.

        Parameters:
            PDE_list (array-like): PDE values.
            SSA_list (array-like): SSA values.

        Returns:
            dict: Contains:
                - 'propensity' (np.ndarray): Propensity values.
                - 'min_mass_SSA' (np.ndarray): Active SSA compartments based on minimum mass.
                - 'min_mass_PDE' (np.ndarray): Active PDE compartments based on minimum mass.
                - 'threshold_SSA' (np.ndarray): Active SSA compartments based on threshold.
                - 'threshold_PDE' (np.ndarray): Active PDE compartments based on threshold.
                - 'combined_mass' (np.ndarray): Combined mass at each compartment.
                - 'approx_PDE_mass' (np.ndarray): PDE contribution.
        """
        PDE_list = np.array(PDE_list, dtype=np.float32)
        SSA_list = np.array(SSA_list, dtype=np.int32)
        propensity = np.zeros(6 * self.SSA_M, dtype=np.float32)

        # Compute boolean masks based on minimum mass
        min_mass_PDE, min_mass_SSA = self.boolean_low_limit(PDE_list)

        # Compute combined mass and approximate PDE mass
        combined_mass, approx_PDE_mass = self.calculate_total_mass(PDE_list, SSA_list)

        # Compute boolean masks based on threshold
        threshold_SSA, threshold_PDE = self.boolean_threshold_mass(combined_mass)

        # Call the C function to calculate propensities
        self.lib.CalculatePropensity(
            ctypes.c_int(self.SSA_M),
            PDE_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            SSA_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            propensity.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            threshold_SSA.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            combined_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            approx_PDE_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            min_mass_SSA.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_float(self.degradation_rate_h),
            ctypes.c_float(self.threshold),
            ctypes.c_float(self.production_rate),
            ctypes.c_float(self.gamma),
            ctypes.c_float(self.jump_rate)
        )

        return {
            "propensity": propensity,
            "min_mass_SSA": min_mass_SSA,
            "min_mass_PDE": min_mass_PDE,
            "threshold_SSA": threshold_SSA,
            "threshold_PDE": threshold_PDE,
            "combined_mass": combined_mass,
            "approx_PDE_mass": approx_PDE_mass,
        }
    
class SSA_C_Wrapper:
    def __init__(self, params, library_path="C_functions.so"):
        """
        Initialize the HybridSimulation with the given parameters.

        Parameters:
            params  (dict): Dictionary containing the following keys:
                - domain_length (float): Length of the spatial domain.
                - compartment_number (int): Number of SSA compartments.
                - total_time (float): Total simulation time.
                - timestep (float): Time step size for numerical integration.
                - degradation_rate (float): Degradation rate for the system.
                - production_rate (float): Production rate for the system.
                - diffusion_rate (float): Diffusion rate for the system.
                - h (float): Grid size or compartment length.
        """
        self.lib = ctypes.CDLL(library_path)


        self.L = params[ 'domain_length']
        self.SSA_M = params[ 'compartment_number']
        self.total_time = params[ 'total_time']
        self.timestep = params[ 'timestep']
        self.degradation_rate = params[ 'degradation_rate']
        self.production_rate = params[ 'production_rate']
        self.diffusion_rate = params[ 'diffusion_rate']
        self.h = params[ 'h']

        self.degradation_rate_h = params[ 'degradation_rate'] / params[ 'h']  # Degradation rate 
        self.jump_rate = self.diffusion_rate / (self.h**2)  # Diffusion coefficient scaled by grid size
        self.lib.CalculatePropensitySSA.argtypes = [
            ctypes.c_int,  # SSA_M
            ctypes.POINTER(ctypes.c_int),  # SSA_list
            ctypes.POINTER(ctypes.c_float),  # propensity_list
            ctypes.c_float,  # degradation_rate_h
            ctypes.c_float,  # production_rate
            ctypes.c_float   # jump_rate
        ]



    def __str__(self):
        return (f"HybridSimulation(domain_length={self.L}, compartments={self.SSA_M}, "
                f"total_time={self.total_time}, timestep={self.timestep}, "
                f"degradation_rate={self.degradation_rate}, production_rate={self.production_rate}, "
                f"diffusion_rate={self.diffusion_rate}, h={self.h}, d={self.d})")
    

    def calculate_propensity(self,SSA_list):
        """
        Calculate the propensity of reactions based on the current SSA state.

        Parameters:
            SSA_list (np.ndarray): Current state of the SSA compartments.

        Returns:
            np.ndarray: Propensity values for each compartment.
        """
        # Placeholder for actual implementation
        # This should call the C function to compute propensities
        # based on the SSA_list and other parameters
        #(int SSA_M, int *SSA_list, float *propensity_list, float degradation_rate_h, float production_rate, float jump_rate)

        propensity = np.zeros(3*self.SSA_M, dtype=np.float32)
        SSA_list = np.array(SSA_list, dtype=np.int32)

        self.lib.CalculatePropensitySSA(
            ctypes.c_int(self.SSA_M),
            SSA_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            propensity.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(self.degradation_rate_h),
            ctypes.c_float(self.production_rate),
            ctypes.c_float(self.jump_rate)
        )

        return {
            "propensity":propensity
        }



class Numerical_wrapper:
    def __init__(self, params, library_path="C_Numerical_functions.so"):
        """
        Initialize the Numerical_wrapper with parameters.

        Parameters:
            params (dict): Dictionary containing the following keys:
                - PDE_M (int): Number of PDE grid points.
                - diffusion_rate (float): Diffusion rate for the PDE system.
                - degradation_rate (float): Degradation rate for the PDE system.
                - production_rate (float): Production rate for the PDE system.
                - deltax (float): Spatial discretization step size.
                - timestep (float): Time step size for numerical integration.
            library_path (str): Path to the compiled C library (default is "C_Numerical_functions.so").
        """
        self.PDE_M = params['PDE_M']  # Number of PDE grid points
        self.diffusion_rate = params['diffusion_rate']  # Diffusion rate for the PDE system
        self.degradation_rate = params['degradation_rate']  # Degradation rate for the PDE system
        self.production_rate = params['production_rate']  # Production rate for the PDE system
        self.deltax = params['deltax']  # Spatial discretization step size
        self.timestep = params['timestep']  # Time step size for numerical integration

        # Create the finite difference matrix for diffusion
        self.DNabla = self.create_finite_diff()

    def create_finite_diff(self):
        """
        Creates the finite difference matrix for the diffusion term.

        Returns:
            np.ndarray: Finite difference matrix scaled by the diffusion coefficient.
        """
        diff_coefficient = self.diffusion_rate / (self.deltax**2)  # Scaling factor for diffusion
        DX = np.zeros((self.PDE_M, self.PDE_M), dtype=int)  # Initialize the finite difference matrix

        # Set boundary conditions
        DX[0, 0], DX[-1, -1] = -1, -1
        DX[0, 1], DX[-1, -2] = 1, 1

        # Fill the interior of the matrix
        for i in range(1, DX.shape[0] - 1):
            DX[i, i] = -2
            DX[i, i + 1] = 1
            DX[i, i - 1] = 1

        return diff_coefficient * DX  # Scale the matrix by the diffusion coefficient

    def _RHS_derivative(self, old_vector, boolean_threshold, SSA_fine_mass):
        """
        Calculates the right-hand side (RHS) of the PDE system.

        Parameters:
            old_vector (np.ndarray): Current state of the PDE grid.
            boolean_threshold (np.ndarray): Boolean mask for active regions.
            SSA_fine_mass (np.ndarray): Fine-grained SSA mass mapped to the PDE grid.

        Returns:
            np.ndarray: The RHS of the PDE system.
        """
        # Compute the diffusion term
        diffusion_term = self.DNabla @ old_vector

        # Compute the degradation term (nonlinear)
        degradation_term = self.degradation_rate * boolean_threshold * ((old_vector + SSA_fine_mass) ** 2)

        # Compute the production term
        production_term = self.production_rate * boolean_threshold * (old_vector + SSA_fine_mass)

        # Combine all terms to compute the RHS
        dudt = diffusion_term - degradation_term + production_term
        return dudt

    def RK4_steps(self, old_vector, boolean_threshold, SSA_fine_mass):
        """
        Perform one step of the Runge-Kutta 4th order (RK4) method for time integration.

        Parameters:
            old_vector (np.ndarray): Current state of the PDE grid.
            boolean_threshold (np.ndarray): Boolean mask for active regions.
            SSA_fine_mass (np.ndarray): Fine-grained SSA mass mapped to the PDE grid.

        Returns:
            np.ndarray: Updated state of the PDE grid after one RK4 step.
        """
        # Compute the RK4 coefficients
        k1 = self._RHS_derivative(old_vector, boolean_threshold, SSA_fine_mass)
        k2 = self._RHS_derivative(old_vector + 0.5 * self.timestep * k1, boolean_threshold, SSA_fine_mass)
        k3 = self._RHS_derivative(old_vector + 0.5 * self.timestep * k2, boolean_threshold, SSA_fine_mass)
        k4 = self._RHS_derivative(old_vector + self.timestep * k3, boolean_threshold, SSA_fine_mass)

        # Combine the coefficients to compute the next state
        return old_vector + self.timestep * (k1 + 2 * k2 + 2 * k3 + k4) / 6