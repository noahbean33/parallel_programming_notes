/*
 Program: Parallel maximum via MPI_Allgather
 Purpose: Each process computes a local max over its chunk; all local maxima
          are gathered on all ranks, and a global max is computed.
 Compile: mpicc -O2 -o parallel_max parallel_max.c
 Run:     mpirun -np <P> ./parallel_max
 Notes:
 - Global problem size N must be divisible by the number of processes.
 - Uses rank-based srand seed so each process generates different data.
 MPI APIs used:
 - MPI_Init, MPI_Comm_size, MPI_Comm_rank, MPI_Finalize
 - MPI_Allgather
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main() {

    int numberP;
    int rank;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&numberP);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    // Define the size of the global array
    int N = 42;

    // Allocate memory for the local array
    int* local_array = (int *)malloc(N/numberP * sizeof(int));

    // Set a specific seed for the RNG based on the rank
    unsigned int seed = rank + 1;
    srand(seed);

    // Fill the local portion of the array with random values
    for (int i = 0; i < N/numberP; i++) {
        local_array[i] = rand() % 100;

    }

    // Print the local array of this process
    printf("Process %d Local Array: ",rank);
    for (int i = 0; i < N/numberP; ++i) {
        printf("%d ", local_array[i]);
    }
    printf("\n");
    // Find Local Maximum
    int local_max = local_array[0];
    for (int i = 1; i < N/numberP; ++i) {
        if (local_array[i] > local_max) {
            local_max = local_array[i];
        }
    }

    // Use MPI_Allgather to gather all local maximum values
    int *all_max_values = (int *)malloc(numberP*sizeof(int));
    MPI_Allgather(&local_max,1,MPI_INT,all_max_values,1,
                  MPI_INT,MPI_COMM_WORLD);

    // Print all local maximum values
    printf("Process %d Local max: %d, All max values: ",rank,local_max);
    for (int i = 0; i < numberP; ++i) {
        printf("%d ",all_max_values[i]);
        if (i == numberP-1) printf("\n");
    }

    // Determine the global maximum
    int global_max = all_max_values[0];
    for (int i = 1; i < numberP; ++i) {
        if (all_max_values[i] > global_max) {
            global_max = all_max_values[i];
        }
    }

    // Display results
    printf("Process %d Global Max: %d\n",rank,global_max);

    // Cleanup
    free(local_array);
    free(all_max_values);

    MPI_Finalize();

    return 0;

}







