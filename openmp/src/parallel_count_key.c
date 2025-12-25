/*
 * This program counts the occurrences of a specific key in an array in parallel using OpenMP taskloop.
 * It demonstrates the use of the reduction clause to safely update the count across multiple tasks.
 *
 * Compilation instructions:
 * gcc -o parallel_count_key -fopenmp Problem1_PointA.c
 */
#include <omp.h>

// This function counts the occurrences of a 'key' in an array 'a' of length 'Nlen' using a parallel taskloop.
long count_iter(long Nlen, long *a, long key) {
    long count = 0;

    // Use a taskloop to parallelize the counting process.
    // The reduction(+:count) clause ensures that the 'count' variable is correctly updated by all tasks.
    #pragma omp taskloop num_tasks(omp_get_num_threads()) reduction(+:count)
    for (int i = 0; i < Nlen; ++i) {
        if (a[i] == key) {
            ++count;
        }
    }
    return count;
