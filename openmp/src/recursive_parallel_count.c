/*
 * This program uses a recursive approach with OpenMP tasks to count the occurrences of a key in an array.
 * The recursion is parallelized up to a certain depth (CUTOFF), after which it proceeds sequentially to avoid excessive task creation.
 *
 * To use this function, it must be called from within an OpenMP parallel region, like so:
 * #pragma omp parallel
 * {
 *     #pragma omp single
 *     total_count = count_recur(N, array, key, 0);
 * }
 *
 * Compilation instructions:
 * gcc -o recursive_parallel_count -fopenmp Problem1_PointB.c
 */
#include <omp.h>

#define CUTOFF 3

// Recursively counts occurrences of 'key' in array 'a'.
// 'Nlen' is the length of the array segment being processed.
// 'depth' tracks the recursion depth to limit task creation.
long count_recur(long Nlen, long *a, long key, int depth) {
    long count1 = 0, count2 = 0;

        // Base case: if the array segment has only one element, check if it's the key.
    if (Nlen == 1) {
        return (a[0] == key) ? 1 : 0;
    } else {
                // If the recursion depth is below the cutoff, create parallel tasks for each half of the array.
        if (depth < CUTOFF) {
                        // Create a task to process the first half of the array.
            #pragma omp task shared(count1)
            count1 = count_recur(Nlen / 2, a, key, depth + 1);

                        // Create a task to process the second half of the array.
            #pragma omp task shared(count2)
            count2 = count_recur(Nlen - Nlen / 2, a + Nlen / 2, key, depth + 1);

                        // Wait for the two tasks to complete before proceeding.
            #pragma omp taskwait
        } else {
            // If the cutoff depth is reached, continue the recursion sequentially without creating new tasks.
            count1 = count_recur(Nlen / 2, a, key, depth + 1);
            count2 = count_recur(Nlen - Nlen / 2, a + Nlen / 2, key, depth + 1);
        }
    }

    return count1 + count2;
}
