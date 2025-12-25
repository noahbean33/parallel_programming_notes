/*
 * This program demonstrates a task-based workflow using OpenMP with dependencies to perform SAXPY operations.
 * It initializes two vectors, runs SAXPY on them twice, and writes the results to files.
 *
 * Note:
 * 1. This code is a snippet and requires an implementation for the 'saxpy' function.
 * 2. The 'depend' syntax on arrays should ideally use array sections (e.g., 'depend(out: fx[0:N])') for clarity and correctness.
 * 3. A '#pragma omp taskwait' is critically needed before the 'free' and 'fclose' calls to prevent a race condition
 *    where the program deallocates resources before the tasks using them are finished.
 *
 * Compilation instructions (assuming a 'saxpy.c' is provided):
 * gcc -o saxpy_task_pipeline -fopenmp problem4.c saxpy.c
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main() {
    int N = 1 << 20; /* 1 million floats */
    float *fx = (float *) malloc(N * sizeof(float));
    float *fy = (float *) malloc(N * sizeof(float));
    FILE *fpx = fopen("fx.out", "w");
    FILE *fpy = fopen("fy.out", "w");
    /* simple initialization just for testing */
    #pragma omp parallel
    #pragma omp single
    
        // Task 1: Initialize the 'fx' array. This task produces 'fx'.
    #pragma omp task depend(out: fx) // Task 1
    for (int k = 0; k < N; ++k)
        fx[k] = 2.0f + (float) k;
        // Task 2: Initialize the 'fy' array. This task produces 'fy'.
    #pragma omp task depend(out: fy) // Task 2
    for (int k = 0; k < N; ++k)
        fy[k] = 1.0f + (float) k;
    /* Run SAXPY TWICE */
		// Task 3: Perform SAXPY(fy = 3.0*fx + fy). Depends on 'fx' from Task 1 and 'fy' from Task 2.
	#pragma omp task depend(in:fx) depend(inout: fy) // Task 3
    saxpy(N, 3.0f, fx, fy);
		// Task 4: Perform SAXPY(fx = 5.0*fy + fx). Depends on 'fy' from Task 3.
	#pragma omp task depend(in:fy) depend(inout: fx) // Task 4
    saxpy(N, 5.0f, fy, fx);
    /* Save results */
	    // Task 5: Write the final 'fx' array to a file. Depends on 'fx' from Task 4.
	#pragma omp task depend(in: fx) // Task 5
    for (int k = 0; k < N; ++k)
        fprintf(fpx, " %f ", fx[k]);
	    // Task 6: Write the final 'fy' array to a file. Depends on 'fy' from Task 3.
	#pragma omp task depend(in: fy) // Task 6
    for (int k = 0; k < N; ++k)
        fprintf(fpy, " %f ", fy[k]);
        // CRITICAL: A taskwait is needed here to ensure all tasks complete before cleanup.
    // #pragma omp taskwait
    free(fx); fclose(fpx);
        free(fy); fclose(fpy);
    return 0;
}
