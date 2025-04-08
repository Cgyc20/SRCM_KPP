#include <stdio.h>

void create_finite_difference(int PDE_M, int *nabla){
    //create the finite difference matrix 
    //for the PDE

    int findex = PDE_M-1; //final index
    nabla[0][0] = -1;
    nabla[findex][findex] = -1;
    nabla[0][1] = 1;
    nabla[findex][findex-1] = 1;

    for (int i=1; i<findex; i++){
        nabla[i][i]=-2;
        nabla[i][i+1]=1;
        nabla[i][i-1]=1;
    }
}

void RHS_deriv(float *old_vector, int *boolean_threshold, float *SSA_fine_mass, int *nabla_diff){
    //This will calculate the RHS of the function of the Fisher KPP equation.
    float diff_term = 
}