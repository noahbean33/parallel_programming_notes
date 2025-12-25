/*
 * This program calculates a frequency histogram for an array of integers in parallel using OpenMP.
 * It divides the input array into chunks, and each thread processes its assigned chunk.
 * An atomic operation is used to prevent race conditions when updating the histogram bins.
 *
 * Note: HIST_SIZE is not defined in this file and must be defined for the code to compile.
 * 
 * Compilation instructions:
 * gcc -o parallel_histogram -fopenmp FindFrequency.c -D HIST_SIZE=<value>
 */
#include <omp.h>

// This function calculates a histogram of an integer array in parallel using OpenMP.
void FindFrequency(int *input, int size,int *histogram, int min, int max) {
        // Determine the chunk size for each thread.
    int chunk_size = (size + omp_get_max_threads() -1) / omp_get_max_threads();
    
        // Start a parallel region.
    # pragma omp parallel num_threads(omp_get_max_threads())
    {
                // Get the thread ID.
        int tid = omp_get_thread_num();
                // Calculate the start and end of the chunk for the current thread.
        int start = tid * chunk_size;
        int end = (tid + 1) * chunk_size;
                // Ensure the chunk does not go beyond the array size.
        if (end > size) end = size;
        
        int tmp;
        
                // Each thread processes its chunk of the array.
        for (int i = start; i < end; ++i) {
                        // Calculate the histogram bin for the current element.
            tmp = (input[i] - min) * (HIST_SIZE / (max - min - 1));
                        // Use an atomic operation to safely increment the histogram bin.
            #pragma omp atomic
            histogram[tmp]++;
        }
    }
}
            
