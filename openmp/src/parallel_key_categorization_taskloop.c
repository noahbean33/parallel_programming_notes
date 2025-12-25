/*
 * This program demonstrates key categorization using an OpenMP taskloop with a fixed number of tasks.
 * It searches for keys in a database and organizes the results into buckets, using locks for thread safety.
 *
 * Note:
 * 1. This file requires external definitions for getkeys(), init_DBin(), clear_DBout(), and clear_counter().
 * 2. The 'nogroup' clause is used, which can be dangerous without a subsequent '#pragma omp taskwait'.
 *    Without it, the program might destroy locks before all tasks are complete. A taskwait should be added after the loop.
 *
 * Compilation instructions (assuming other necessary source files are provided):
 * gcc -o parallel_key_categorization_taskloop -fopenmp Problem2PointB.c other_source_files.c
 */
#include <omp.h>
#define DBsize 1048576
#define nkeys 16 // the number of processors can be larger than the number of keys

int main() {
    // DBin: Input database, DBout: Output categorized database, keys: Search keys, counter: Tracks counts for each key.
	double keys[nkeys], DBin[DBsize], DBout[nkeys][DBsize];
	unsigned int i, k, counter[nkeys];

		// Initialize data structures (definitions are in a separate file).
	getkeys(keys, nkeys);             // get keys
	init_DBin(DBin, DBsize);          // initialize elements in DBin
	clear_DBout(DBout, nkeys, DBsize); // initialize elements in DBout
	clear_counter(counter, nkeys);     // initialize counter to zero

		// Declare and initialize an array of locks, one for each key.
	omp_lock_t locks[nkeys];
	for (i = 0; i < nkeys; ++i)
		omp_init_lock(&locks[i]);

		// Create a parallel region. A single thread will set up the tasks.
	#pragma omp parallel
	#pragma omp single
		// Use a taskloop to create 4 tasks to process the loop. 'nogroup' removes the implicit barrier at the end of the loop.
	// A '#pragma omp taskwait' should be placed after this loop to ensure all tasks are finished before destroying the locks.
	#pragma omp taskloop firstprivate(i) num_tasks(4) nogroup
	for (i = 0; i < DBsize; ++i) {
		for (k = 0; k < nkeys; ++k) {
							if (DBin[i] == keys[k]) {
					// If a match is found, lock the corresponding bucket to ensure a thread-safe update.
					omp_set_lock(&locks[k]);
					DBout[k][counter[k]++] = i;
					omp_unset_lock(&locks[k]);
				}
				omp_set_lock(&locks[k]);
											}
		}
	}

		// Destroy the locks to release resources.
	for (i = 0; i < nkeys; ++i)
		omp_destroy_lock(&locks[i]);

	return 0;
}
