/*
 * This program categorizes elements from a database into different buckets based on a set of keys.
 * It uses an OpenMP taskloop to parallelize the search and locks to ensure thread-safe updates to the output database.
 *
 * Note: This file requires external definitions for getkeys(), init_DBin(), clear_DBout(), and clear_counter().
 *
 * Compilation instructions (assuming other necessary source files are provided):
 * gcc -o parallel_key_categorization -fopenmp Problem2PointA.c other_source_files.c
 */
#include <omp.h>
#define DBsize 1048576
#define nkeys 16 // the number of processors can be larger than the number of keys
int main() {
    // DBin: Input database, DBout: Output categorized database, keys: Search keys, counter: Tracks counts for each key.

double keys[nkeys], DBin[DBsize], DBout[nkeys][DBsize];
unsigned int i, k, counter[nkeys];
    // Initialize data structures (definitions are in a separate file).
    getkeys(keys, nkeys); // get keys
init_DBin(DBin, DBsize); // initialize elements in DBin
clear_DBout(DBout, nkeys, DBsize); // initialize elements in DBout
clear_counter(counter, nkeys); // initialize counter to zero

    // Create a parallel region, but have only a single thread set up the tasks.
    #pragma omp parallel
#pragma omp single

    // Declare an array of locks, one for each key, to manage concurrent access.
    omp_lock_t locks[nkeys];

    // Initialize all the locks before using them.
    for (i = 0; i < nkeys; ++i) omp_init_lock(&locks[i]);
	
	
    // Use a taskloop to distribute the iterations of the outer loop among tasks.
    // 'k' is made private to each task to avoid data races.
    #pragma omp taskloop private(k)
for (i = 0; i < DBsize; ++i)
	for(k = 0; k < nkeys; ++k)
		        if(DBin[i] == keys[k]) {
            // If a match is found, lock the corresponding key's bucket to prevent race conditions.
            omp_set_lock(&locks[k]);
            // Add the index 'i' to the correct bucket and increment its counter.
            DBout[k][counter[k]++] = i;
            // Release the lock.
            omp_unset_lock(&locks[k]);
        }

    // Destroy the locks to release resources.
    for (i = 0; i < nkeys; ++i) omp_destroy_lock(&locks[i]);	

}

/*
DBout

1 [3,   ....  ]
2 [5,6, ...   ]
3 [           ]
4 [           ]
5 [           ]

/*















