/*
 Program: Parallel sum with Scatter/Gather
 Purpose: Rank 0 initializes an array of size N, scatters equal chunks to all ranks;
          each rank computes a local sum; rank 0 gathers local sums and computes global sum.
 Compile: mpicc -O2 -o parallel_sum parallel_sum.c
 Run:     mpirun -np 4 ./parallel_sum
 Notes:
 - This code declares local_array[N/4]; therefore it assumes np == 4 at runtime.
   Use -np 4 when running (or adjust code accordingly).
 - N must be divisible by np.
 MPI APIs used:
 - MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_Finalize
 - MPI_Scatter, MPI_Gather
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100

int main(int argc, char* argv[]) {
    int id, np;
    int global_array[N];
    int local_array[N/4]; // Each process receives 1/4th of the array
    int local_sum = 0;
    int global_sum = 0;


    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&np);

    // Initialize the global array in process 0
    if (id == 0) {

        for (int i = 0; i < N; ++i) {
            global_array[i] = i + 1;
        }
    }

    // Scatter the global array to local arrays in each process
    MPI_Scatter(global_array,N/np,MPI_INT,local_array,N/np,
                MPI_INT,0,MPI_COMM_WORLD);

    // Calculate local sum
    for (int i = 0; i < N/np; ++i) {
        local_sum += local_array[i];
    }

    // Gather local sums to the global array on process 0
    MPI_Gather(&local_sum,1,MPI_INT,global_array,1,
               MPI_INT,0,MPI_COMM_WORLD);

    // Process 0 prints the result
    if (id == 0) {
        // Calculate the final global sum from the gathered data
        for (int i = 0; i < np; ++i) {
            global_sum += global_array[i];
        }
        printf("Global sum: %d\n",global_sum);
    }


    MPI_Finalize();
    return 0;

}
