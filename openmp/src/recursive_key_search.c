/*
 * This program demonstrates a recursive approach to searching for keys in a database using OpenMP tasks.
 * It recursively splits the database into smaller chunks and processes them in parallel.
 * Locks are used to prevent race conditions when updating the output database.
 *
 * Compilation instructions:
 * gcc -o recursive_key_search -fopenmp Problem2PointC.c
 */
#include <omp.h>
#define DBsize 1048576
#define nkeys 16 // the number of processors can be larger than the number of keys

#define CUT_SIZE 3

omp_lock_t locks[nkeys]

int main() {
    // Define the main data structures for the program.
    // keys: an array to store the keys to search for.
    // DBin: the input database to be processed.
    // DBout: a 2D array to store the categorized results.
    // counter: an array to keep track of the number of elements for each key in DBout.
double keys[nkeys], DBin[DBsize], DBout[nkeys][DBsize];
unsigned int i, k, counter[nkeys];
    // Initialize the data structures.
    getkeys(keys, nkeys); // get keys
init_DBin(DBin, DBsize); // initialize elements in DBin
clear_DBout(DBout, nkeys, DBsize); // initialize elements in DBout
clear_counter(counter, nkeys); // initialize counter to zero

    // Create a parallel region to execute the tasks.
    #pragma omp parallel
#pragma omp single

    // Initialize a lock for each key to prevent race conditions during updates.
    for (i = 0; i < nkeys; ++i) omp_init_lock(&locks[i]);

    // Start the recursive key search process.
    rec_keys(DBin,DBout,0,DBsize-1,counter);

    // Destroy the locks after the processing is complete.
    for (i = 0; i < nkeys; ++i) omp_destroy_lock(&locks[i]);	

}


// This function recursively searches for keys in a portion of the database (from ini to end).
void rec_keys(double *DBin, double *DBout, int ini, int end,
	unsigned int *counter) {
	
	    // Base case for the recursion: if the chunk size is 1, process the element.
	if (end - ini + 1) <= 1) { // Base Case
		for (unsigned int k = 0; k < nkeys; ++k) {
			            // If the database element matches a key, add it to the corresponding output bucket.
			if (DBin[ini] == keys[k]){
				                // Lock to ensure thread-safe access to the shared DBout and counter arrays.
				omp_set_lock(&locks[k]);
								DBout[k][counter[k]++] = i; // Note: 'i' is not defined in this scope, which may lead to unexpected behavior.
				                // Unlock after the update is complete.
				omp_unset_lock(&locks[k]);
				
		}
	}
		else { // Recursive Case: if the chunk size is larger than 1, split it.

		
		        // Calculate the midpoint to split the current chunk into two halves.
		int half = (end - ini + 1)/2;
		        // If the current task is not a 'final' task, create new sub-tasks.
		if (!omp_in_final()) { // If task is not final:
            // Create a new task for the first half. The task becomes a 'final' task if its size is below or equal to CUT_SIZE.
			#pragma omp task final (half <= CUT_SIZE)
			rec_keys(DBin,DBout,ini,half,counter);
            // Create a new task for the second half.
			#pragma omp task final (half <= CUT_SIZE)
			rec_keys(DBin,DBout,half+1,end,counter);
		}
				else { // If the current task is 'final', execute the recursion sequentially without creating new tasks.

			rec_keys(DBin,DBout,ini,half,counter);
			rec_keys(DBin,DBout,half+1,end,counter);
		}
	}
}
		
	
	
	
/*	
			
	CUT_SIZE = 2		
			
		ini                   end	
	DBin[v1,v2,v3,v4,v5,v6,v7,v8]
			      Task 1			Task 2
			 ini      half     half+1    end
			[v1,v2,v3,v4]     [v5,v6,v7,v8]
		[v1,v2] [v3,v4]    [v5,v6] [v7,v8]
	 [v1] [v2] [v3]  [v4] [v5] [v6] [v7] [v8]

*/
	 
	 
	 
	 
	 
	 
	 
	 
