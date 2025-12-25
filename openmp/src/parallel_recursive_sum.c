/*
 * This program calculates the sum of a vector's elements using a recursive approach with OpenMP tasks.
 *
 * Note:
 * 1. This function must be called from within an active parallel region (e.g., inside a '#pragma omp parallel' block)
 *    for the tasks to execute in parallel.
 * 2. The current implementation is inefficient. It creates a single task that performs both recursive calls sequentially.
 *    A better approach would be to create two separate tasks for each recursive call to enable true parallel execution.
 *
 * Compilation instructions:
 * gcc -o parallel_recursive_sum -fopenmp recursive_sum_vector.c
 */
#include <omp.h>

// This function recursively calculates the sum of the first 'n' elements of vector 'X'.
int recursive_sum_vector(int *X, int n) {
    int sum = 0;
        // Base case: If the vector has only one element, return that element.
    if (n == 1) { return X[0]; }
    else {
                // A parallel region is created here, but the #pragma omp single ensures only one thread proceeds.
        #pragma omp parallel
        {
            #pragma omp single
            {
                                // A single task is created to perform the recursive summation.
                // The two recursive calls inside this task are executed sequentially, not in parallel.
                #pragma omp task shared(sum)
                {
                    int half = n / 2;
                                        // Recursively sum the two halves of the vector.
                    sum = recursive_sum_vector(X,half) + recursive_sum_vector(X + half,n-half);
                }
            }
        }
    }
    return sum;
}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                    
    
    
