/*
 * This program uses OpenMP to find the minimum and maximum values in an integer array in parallel.
 * It divides the array into chunks, and each thread finds the local min and max in its chunk.
 * A critical section is used to safely update the global minimum and maximum values.
 *
 * Compilation instructions:
 * gcc -o parallel_find_bounds -fopenmp FindBounds.c
 */
#include <omp.h>

// This function finds the minimum and maximum values in an array in parallel using OpenMP.
void FindBounds(int *input, int size, int *min, int *max) {
    
        // Calculate the size of the chunk that each thread will process.
    int chunk_size = (size + omp_get_max_threads()-1)/ omp_get_max_threads();
        // Initialize local min and max with the first element of the array.
    int local_min = input[0], local_max = input[0];
    
        // Start a parallel region with the maximum number of available threads.
    #pragma omp parallel num_threads(omp_get_max_threads())
    
        // Get the thread ID.
    int tid = omp_get_thread_num();
        // Determine the start and end of the chunk for the current thread.
    int start = tid * chunk_size;
    int end = (tid+1) * chunk_size;
        // Ensure the end of the chunk does not exceed the array size.
    if (end > size) end = size;
    
        // Each thread finds the local min and max in its assigned chunk.
    for (int i = start; i < end; ++i) {
                if (input[i] > local_max) local_max = input[i];
                if (input[i] < local_min) local_min = input[i];
    
    }
    // To avoid race conditions when updating the global minimum and maximum values, we use a critical section (#pragma omp critical) to ensure mutual exclusion.
    
        // Use a critical section to safely update the global min and max values.
    #pragma omp critical {
    
        if (local_max > *max) *max = local_max;
        if (local_min < *min) *min = local_min;
    }
    
}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
