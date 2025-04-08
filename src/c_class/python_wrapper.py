import numpy as np
import ctypes

class CFunctionWrapper:
    def __init__(self, library_path="C_functions.so"):
        """
        Initialize the wrapper by loading the C library.
        """
        self.lib = ctypes.CDLL(library_path)

        # Define argument types for the C functions
        self.lib.ApproximateMassLeftHand.argtypes = [
            ctypes.c_int,  # SSA_M
            ctypes.c_int,  # PDE_multiple
            ctypes.POINTER(ctypes.c_float),  # PDE_list
            ctypes.POINTER(ctypes.c_float),  # approxMass
            ctypes.c_float  # deltax
        ]

        self.lib.BooleanMass.argtypes = [
            ctypes.c_int,  # SSA_m
            ctypes.c_int,  # PDE_m
            ctypes.c_int,  # PDE_multiple
            ctypes.POINTER(ctypes.c_float),  # PDE_list
            ctypes.POINTER(ctypes.c_int),  # boolean_PDE_list
            ctypes.POINTER(ctypes.c_int),  # boolean_SSA_list
            ctypes.c_float  # h
        ]

        self.lib.BooleanThresholdMass.argtypes = [
            ctypes.c_int,  # SSA_m
            ctypes.c_int,  # PDE_m
            ctypes.c_int,  # PDE_multiple
            ctypes.POINTER(ctypes.c_float),  # combined_list
            ctypes.c_float,  # h
            ctypes.POINTER(ctypes.c_int),  # compartment_bool_list
            ctypes.POINTER(ctypes.c_int),  # PDE_bool_list
            ctypes.c_float  # threshold
        ]

        self.lib.FineGridSSAMass.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # SSA_mass
            ctypes.c_int,  # PDE_grid_length
            ctypes.c_int,  # SSA_m
            ctypes.c_int,  # PDE_multiple
            ctypes.c_float,  # h
            ctypes.POINTER(ctypes.c_float)  # fine_SSA_Mass
        ]

        self.lib.CalculatePropensity.argtypes = [
            ctypes.c_int,  # SSA_M
            ctypes.POINTER(ctypes.c_float),  # PDE_list
            ctypes.POINTER(ctypes.c_int),  # SSA_list
            ctypes.POINTER(ctypes.c_float),  # propensity_list
            ctypes.POINTER(ctypes.c_int),  # boolean_SSA_list
            ctypes.POINTER(ctypes.c_float),  # combined_mass_list
            ctypes.POINTER(ctypes.c_float),  # Approximate_PDE_Mass
            ctypes.POINTER(ctypes.c_int),  # boolean_mass_list
            ctypes.c_float,  # degradation_rate_h
            ctypes.c_float,  # threshold
            ctypes.c_float,  # production_rate
            ctypes.c_float,  # gamma
            ctypes.c_float  # jump_rate
        ]

        print(f"Loaded C library from {library_path}")

    def __str__(self):
        return f"CFunctionWrapper(library_path={self.lib._name}) for the Hybrid method"
    

    def approximate_mass_left_hand(self, SSA_M, PDE_multiple, PDE_list, deltax):
        """
        Computes the approximate PDE mass at each SSA compartment (left-hand weighted sum).
        
        Parameters:
            SSA_M (int): Number of SSA compartments.
            PDE_multiple (int): Number of PDE points per SSA compartment.
            PDE_list (array-like): Values of the PDE grid.
            deltax (float): Spatial discretization step size.

        Returns:
            np.ndarray: Approximate PDE mass in each SSA compartment.
        """
        approximate_PDE_mass = np.zeros(SSA_M, dtype=np.float32)
        PDE_list = np.array(PDE_list, dtype=np.float32)

        self.lib.ApproximateMassLeftHand(
            ctypes.c_int(SSA_M),
            ctypes.c_int(PDE_multiple),
            PDE_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            approximate_PDE_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(deltax)
        )

        return approximate_PDE_mass

    def calculate_total_mass(self, PDE_list, SSA_list, PDE_multiple, deltax, SSA_M):
        """
        Computes the total combined mass of PDE and SSA at each SSA compartment.

        Parameters:
            PDE_list (array-like): PDE values across the domain.
            SSA_list (array-like): SSA values per compartment.
            PDE_multiple (int): Number of PDE grid points per SSA compartment.
            deltax (float): Spatial step.
            SSA_M (int): Number of SSA compartments.

        Returns:
            tuple:
                combined_list (np.ndarray): Total mass per compartment (PDE + SSA).
                approximate_PDE_mass (np.ndarray): The approximate PDE mass component.
        """


        PDE_list = np.array(PDE_list, dtype=np.float32)
        SSA_list = np.array(SSA_list, dtype=np.int32)

        approximate_PDE_mass = self.approximate_mass_left_hand(SSA_M, PDE_multiple, PDE_list, deltax)
        combined_list = np.add(SSA_list, approximate_PDE_mass)

        return combined_list, approximate_PDE_mass

    def boolean_mass(self, SSA_m, PDE_m, PDE_multiple, PDE_list, h):
        """
        Computes boolean masks (0 or 1) for where there is significant PDE and SSA mass.

        Parameters:
            SSA_m (int): Number of SSA compartments.
            PDE_m (int): Number of PDE points.
            PDE_multiple (int): PDE points per SSA compartment.
            PDE_list (array-like): PDE concentration values.
            h (float): Grid size or thresholding parameter.

        Returns:
            tuple:
                boolean_PDE_list (np.ndarray): Mask for PDE domain.
                boolean_SSA_list (np.ndarray): Mask for SSA compartments.
        """

        PDE_list = np.array(PDE_list, dtype=np.float32)
        boolean_PDE_list = np.zeros(PDE_m, dtype=np.int32)
        boolean_SSA_list = np.zeros(SSA_m, dtype=np.int32)

        self.lib.BooleanMass(
            ctypes.c_int(SSA_m),
            ctypes.c_int(PDE_m),
            ctypes.c_int(PDE_multiple),
            PDE_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            boolean_PDE_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            boolean_SSA_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_float(h)
        )

        return boolean_PDE_list, boolean_SSA_list

    def boolean_threshold_mass(self, SSA_m, PDE_m, PDE_multiple, combined_list, h, threshold):
        """
        Computes boolean masks based on a threshold for total (PDE + SSA) mass.

        Parameters:
            SSA_m (int): Number of SSA compartments.
            PDE_m (int): Number of PDE points.
            PDE_multiple (int): PDE points per SSA compartment.
            combined_list (array-like): Combined SSA and PDE mass values.
            h (float): Grid size.
            threshold (float): Minimum mass threshold to consider "significant".

        Returns:
            tuple:
                compartment_bool_list (np.ndarray): Mask for SSA compartments.
                PDE_bool_list (np.ndarray): Mask for PDE domain.
        """
        combined_list = np.array(combined_list, dtype=np.float32)
        compartment_bool_list = np.zeros(SSA_m, dtype=np.int32)
        PDE_bool_list = np.zeros(PDE_m, dtype=np.int32)

        self.lib.BooleanThresholdMass(
            ctypes.c_int(SSA_m),
            ctypes.c_int(PDE_m),
            ctypes.c_int(PDE_multiple),
            combined_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(h),
            compartment_bool_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            PDE_bool_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_float(threshold)
        )

        return compartment_bool_list, PDE_bool_list

    def fine_grid_ssa_mass(self, SSA_mass, PDE_grid_length, SSA_m, PDE_multiple, h):
        """
        Spreads SSA mass from coarse compartments into a fine PDE grid representation.

        Parameters:
            SSA_mass (array-like): SSA values per compartment.
            PDE_grid_length (int): Total length of the PDE grid.
            SSA_m (int): Number of SSA compartments.
            PDE_multiple (int): PDE points per SSA compartment.
            h (float): Grid spacing.

        Returns:
            np.ndarray: Fine-grained representation of SSA mass on PDE grid.
        """
        SSA_mass = np.array(SSA_mass, dtype=np.int32)
        fine_SSA_Mass = np.zeros(PDE_grid_length, dtype=np.float32)

        self.lib.FineGridSSAMass(
            SSA_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(PDE_grid_length),
            ctypes.c_int(SSA_m),
            ctypes.c_int(PDE_multiple),
            ctypes.c_float(h),
            fine_SSA_Mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        return fine_SSA_Mass

    def calculate_propensity(self, SSA_M,PDE_M, PDE_multiple, PDE_list, SSA_list, degradation_rate_h, threshold, production_rate, gamma, jump_rate, h, deltax):
            """
            Computes the reaction propensities in each SSA compartment, using PDE influence and other parameters.

            Parameters:
                SSA_M (int): Number of SSA compartments.
                PDE_list (array-like): PDE values.
                SSA_list (array-like): SSA values.
                degradation_rate_h (float): Degradation rate (scaled by h).
                threshold (float): Activation threshold.
                production_rate (float): Rate of production.
                gamma (float): Nonlinear scaling or interaction parameter.
                jump_rate (float): Rate of stochastic jumps.
                h (float): Grid spacing.

            Returns:
                dict: Contains:
                    - 'propensity_list' (np.ndarray): Propensity values.
                    - 'boolean_SSA_list' (np.ndarray): Active SSA compartments, controlling production.
                    - 'combined_mass_list' (np.ndarray): Combined mass at each compartment.
                    - 'approximate_PDE_mass' (np.ndarray): PDE contribution.
                    - 'boolean_mass_list' (np.ndarray): Boolean mask where mass is present.
            """
            # Debug: Print sizes of inputs
          
            assert len(PDE_list) == SSA_M*PDE_multiple, f"Not the right length"

            # Before C function call, enforce:
            PDE_list = np.ascontiguousarray(PDE_list, dtype=np.float32)  # Must match C's float
            SSA_list = np.ascontiguousarray(SSA_list, dtype=np.int32)    # Must match C's int

            # Output array must be pre-allocated with exact size
            propensity_list = np.zeros(6 * SSA_M, dtype=np.float32)  # Explicit initialization

            # Correctly call the instance method with 'self' and pass 'h'
            boolean_SSA_list, Boolean_PDE_list = self.boolean_mass(SSA_M, PDE_M, PDE_multiple, PDE_list, h)

            print(f"Boolean_SSA_list in python is: {boolean_SSA_list}")

            # Debug: Print sizes of intermediate results
            print(f"boolean_SSA_list size: {len(boolean_SSA_list)}")

            # Correctly call the instance method with 'self'
            combined_mass_list, approximate_PDE_mass = self.calculate_total_mass(PDE_list, SSA_list, PDE_multiple, deltax, SSA_M)

            combined_mass_list = np.array(combined_mass_list, dtype=np.float32)

            # Debug: Print sizes of combined mass and approximate PDE mass
            print(f"combined_mass_list size: {len(combined_mass_list)}")
            print(f"approximate_PDE_mass size: {len(approximate_PDE_mass)}")

            # Correctly call the instance method with 'self'
            _, boolean_mass_list = self.boolean_threshold_mass(SSA_M, PDE_M, PDE_multiple, combined_mass_list, h, threshold)

            # Debug: Print size of boolean_mass_list
            print(f"boolean_mass_list size: {len(boolean_mass_list)}")

            self.lib.CalculatePropensity(
                ctypes.c_int(SSA_M),
                PDE_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                SSA_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                propensity_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                boolean_SSA_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                combined_mass_list.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                approximate_PDE_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                boolean_mass_list.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.c_float(degradation_rate_h),
                ctypes.c_float(threshold),
                ctypes.c_float(production_rate),
                ctypes.c_float(gamma),
                ctypes.c_float(jump_rate)
            )

            return {
                "propensity_list": propensity_list,
                "boolean_SSA_list": boolean_SSA_list,
                "boolean_PDE_list": Boolean_PDE_list,
                "combined_mass_list": combined_mass_list,
                "approximate_PDE_mass": approximate_PDE_mass,
                "boolean_mass_list": boolean_mass_list
            }