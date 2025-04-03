import numpy as np
import ctypes

class CFunctionWrapper:
    def __init__(self, library_path="src/c_class/C_functions.so"):
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

    def approximate_mass_left_hand(self, SSA_M, PDE_multiple, PDE_list, deltax):
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
        PDE_list = np.array(PDE_list, dtype=np.float32)
        SSA_list = np.array(SSA_list, dtype=np.int32)

        approximate_PDE_mass = self.approximate_mass_left_hand(SSA_M, PDE_multiple, PDE_list, deltax)
        combined_list = np.add(SSA_list, approximate_PDE_mass)

        return combined_list, approximate_PDE_mass

    def boolean_mass(self, SSA_m, PDE_m, PDE_multiple, PDE_list, h):
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

    def calculate_propensity(self, SSA_M, PDE_list, SSA_list, degradation_rate_h, threshold, production_rate, gamma, jump_rate):
        PDE_list = np.array(PDE_list, dtype=np.float32)
        SSA_list = np.array(SSA_list, dtype=np.int32)
        propensity_list = np.zeros(SSA_M, dtype=np.float32)
        boolean_SSA_list = np.zeros(SSA_M, dtype=np.int32)
        combined_mass_list = np.zeros(SSA_M, dtype=np.float32)
        approximate_PDE_mass = np.zeros(SSA_M, dtype=np.float32)
        boolean_mass_list = np.zeros(SSA_M, dtype=np.int32)

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
            "combined_mass_list": combined_mass_list,
            "approximate_PDE_mass": approximate_PDE_mass,
            "boolean_mass_list": boolean_mass_list
        }