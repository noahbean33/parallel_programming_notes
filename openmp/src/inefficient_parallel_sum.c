/*
 * This program calculates the sum of a vector's elements using a highly inefficient, nested task-based approach with OpenMP.
 *
 * Note:
 * This implementation is for demonstration purposes and is extremely inefficient due to:
 * 1. A taskloop with grainsize(1), which creates a separate task for each element, causing massive overhead.
 * 2. An unnecessary nested '#pragma omp task' inside the taskloop.
 * A much more efficient way to achieve this is with a parallel for and a reduction clause, for example:
 *   #pragma omp parallel for reduction(+:sum)
 *   for (int i = 0; i < n; i++) {
 *       sum += X[i];
 *   }
 *
 * Compilation instructions:
 * gcc -o inefficient_parallel_sum -fopenmp sum_vector.c
 */
#include <omp.h>

// This function calculates the sum of a vector's elements using a highly inefficient nested task structure.
int sum_vector(int *X, int n) {
    int sum = 0;
    
    // A parallel region is created, but only a single thread will generate tasks.
    #pragma omp parallel
    {
        #pragma omp single
        {
                        // A taskloop with grainsize(1) creates a separate task for each iteration of the loop.
            #pragma omp taskloop grainsize(1)
            for (int i = 0; i < n; i++) {
                                // Inside each task from the taskloop, another task is created, which is redundant and inefficient.
                #pragma omp task shared(sum)
                {
                                        // An atomic operation is used to safely add the element to the shared sum.
                    #pragma omp atomic
                    sum += X[i];
                }
            }
        }
    }
    return sum;
}













