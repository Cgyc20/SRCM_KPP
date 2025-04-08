#include <stdio.h>

// Approximate the PDE mass over each SSA compartment using the left-hand rule
void ApproximateMassLeftHand(int SSA_M, int PDE_multiple, float *PDE_list, float *approxMass, float deltax) {
    int start_index, end_index;
    float sum_value;

    for (int i = 0; i < SSA_M; i++) {
        start_index = PDE_multiple * i;           // First PDE index in current SSA compartment
        end_index = PDE_multiple * (i + 1);       // One-past-the-last PDE index for this SSA compartment
        sum_value = 0.0;                          // Reset sum for this compartment

        for (int j = start_index; j < end_index; j++) {
            sum_value += PDE_list[j];             // Accumulate PDE values within the compartment
        }

        approxMass[i] = sum_value * deltax;       // Multiply by grid spacing to approximate mass
    }
}

// Create boolean masks for SSA and PDE domains based on mass threshold
void BooleanMass(int SSA_m, int PDE_m, int PDE_multiple, float *PDE_list, int *boolean_PDE_list, int *boolean_SSA_list, float h) {
    int start_index;
    int BOOL_value;
    int current_index;

    for (int i = 0; i < PDE_m; i++) {
        boolean_PDE_list[i] = (PDE_list[i] > 1 / h) ? 1 : 0;  // Mark 1 if PDE mass at i > 1/h
    }

    for (int i = 0; i < SSA_m; i++) {
        start_index = i * PDE_multiple;          // Start index of PDEs for this SSA block
        BOOL_value = 1;                          // Start with 1, set to 0 if any false in range

        for (int j = 0; j < PDE_multiple; j++) {
            current_index = start_index + j;
            if (current_index >= PDE_m || boolean_PDE_list[current_index] == 0) {
                BOOL_value = 0;                  // If any value in PDE block is false, set whole SSA block false
                break;
            }
        }

        boolean_SSA_list[i] = BOOL_value;        // Assign result for this SSA compartment
    }
}

// Determine whether each compartment and its PDE elements are below a threshold
void BooleanThresholdMass(int SSA_m, int PDE_m, int PDE_multiple, float *combined_list, float h, int *compartment_bool_list, int *PDE_bool_list, float threshold) {
    for (int i = 0; i < SSA_m; i++) {
        compartment_bool_list[i] = (combined_list[i] > threshold) ? 0 : 1;  // 1 if below threshold, else 0
    }

    for (int i = 0; i < SSA_m; i++) {
        int value = compartment_bool_list[i];                 // Boolean flag for this compartment
        int new_value = (value == 0) ? 1 : 0;                 // Invert value
        int start_index = i * PDE_multiple;                   // Start index of PDE sub-block

        for (int j = 0; j < PDE_multiple; j++) {
            PDE_bool_list[start_index + j] = new_value;       // Apply inverted value to each PDE point
        }
    }
}

// Upsample SSA mass to PDE grid for combined dynamics
void FineGridSSAMass(int *SSA_mass, int PDE_grid_length, int SSA_m, int PDE_multiple, float h, float *fine_SSA_Mass) {
    for (int i = 0; i < SSA_m; i++) {
        int start_index = i * PDE_multiple;

        for (int j = 0; j < PDE_multiple; j++) {
            fine_SSA_Mass[start_index + j] = (float)SSA_mass[i] / h;  // Distribute SSA mass evenly across PDE block
        }
    }
}

// Compute all reaction/diffusion propensities for SSA compartments
void CalculatePropensity(int SSA_M, float *PDE_list, int *SSA_list, float *propensity_list, 
                         int *boolean_SSA_list, float *combined_mass_list, 
                         float *Approximate_PDE_Mass, int *boolean_mass_list, 
                         float degradation_rate_h, float threshold, 
                         float production_rate, float gamma, float jump_rate) {

    if (!PDE_list || !SSA_list || !propensity_list || !boolean_SSA_list || 
        !combined_mass_list || !Approximate_PDE_Mass || !boolean_mass_list) {
        fprintf(stderr, "Error: Null pointer passed to CalculatePropensity\n");
        return;
    }

    // printf("Entering CalculatePropensity with SSA_M=%d\n", SSA_M);
    // printf("Pointer addresses:\n");
    // printf("PDE_list: %p\n", (void*)PDE_list);
    // printf("SSA_list: %p\n", (void*)SSA_list);
    // printf("propensity_list: %p\n", (void*)propensity_list);        

    int two_SSA_M = 2 * SSA_M;
    int three_SSA_M = 3 * SSA_M;
    int four_SSA_M = 4 * SSA_M;
    int five_SSA_M = 5 * SSA_M;
    int six_SSA_M = 6 * SSA_M;

    //printf("SSA_M: %d\n", SSA_M);
    //printf("Expected combined_mass_list size: %d\n", SSA_M);
    //printf("Expected propensity_list size: %d\n", 6 * SSA_M);   

    for (int i = 0; i < six_SSA_M; i++) {
        propensity_list[i] = 0.0f;  // Clear all propensity values before computing
    }

    // 1. Diffusion (D → D shift)
    propensity_list[0] = jump_rate * SSA_list[0];  // Left boundary
    for (int i = 1; i < SSA_M - 1; i++) {
        propensity_list[i] = 2.0f * jump_rate * SSA_list[i];  // Middle compartments
    }
    propensity_list[SSA_M - 1] = jump_rate * SSA_list[SSA_M - 1];  // Right boundary

    // 2. Production (D → 2D)
    for (int i = SSA_M; i < two_SSA_M; i++) {
        int index = i - SSA_M;
        if (index < 0 || index >= SSA_M) {
            fprintf(stderr, "Error: index out of bounds. index=%d, SSA_M=%d\n", index, SSA_M);
            return;
        }
        //printf("index: %d\n",index);
        //printf("combined_mass_list: %f\n",combined_mass_list[index]);
        //printf("boolean_SSA_list: %d\n",boolean_SSA_list[index]);

        propensity_list[i] = production_rate * combined_mass_list[index] * boolean_SSA_list[index];  // If SSA is “active”
        //printf("propensity list: %f\n",propensity_list[i]);
    }

    // 3. Degradation: D + D → D (self-degradation)
    for (int i = two_SSA_M; i < three_SSA_M; i++) {
        int index = i - two_SSA_M;
        propensity_list[i] = degradation_rate_h * SSA_list[index] * (SSA_list[index] - 1);  // Quadratic in particle count
    }

    // 4. Degradation: D + C → C (heterogeneous)
    for (int i = three_SSA_M; i < four_SSA_M; i++) {
        int index = i - three_SSA_M;
        propensity_list[i] = 2.0f * degradation_rate_h * SSA_list[index] * Approximate_PDE_Mass[index];  // Interaction with PDE domain
    }

    // 5. Conversion C → D (only when below threshold)
    for (int i = four_SSA_M; i < five_SSA_M; i++) {
        int index = i - four_SSA_M;
        float combined_mass = combined_mass_list[index];
        float approx_mass = Approximate_PDE_Mass[index];
        int boolean_mass = boolean_mass_list[index];

        propensity_list[i] = (combined_mass < threshold) 
                              ? gamma * approx_mass * boolean_mass     // Conversion only allowed below threshold
                              : 0.0f;
    }

    // 6. Conversion D → C (only when above threshold)
    for (int i = five_SSA_M; i < six_SSA_M; i++) {
        int index = i - five_SSA_M;
        //printf("index: %d\n",index);
        //printf("i in propensity: %d\n",i);
        float combined_mass = combined_mass_list[index];
        int SSA_mass = SSA_list[index];

        propensity_list[i] = (combined_mass >= threshold) 
                              ? gamma * SSA_mass                       // Conversion only allowed above threshold
                              : 0.0f;
    }
}