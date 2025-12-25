/*
 Program: Distributed average with MPI_Reduce
 Purpose: Each process creates a local array of random floats, computes its local sum,
          and rank 0 reduces all local sums to compute the global sum and average.
 Compile: mpicc -O2 -o average average.c
 Run:     mpirun -np <P> ./average <num_elements_per_process>
 Notes:
 - Pass the number of elements each process should generate via argv[1].
 - Uses a rank-based srand seed so each process gets different numbers.
 MPI APIs used:
 - MPI_Init, MPI_Comm_rank, MPI_Comm_size
 - MPI_Reduce (sum local sums to rank 0)
 - MPI_Barrier (synchronization before finalize)
 - MPI_Finalize
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
float *create_random_numbers(int num_elements) {

    float *random_numbers = (float *)malloc(sizeof(float)*num_elements);
    for (int i = 0; i < num_elements; ++i) {
        random_numbers[i] = (rand() / (float)RAND_MAX);
    }
    return random_numbers;
}



int main(int argc, char* argv[]) {

    int num_elems_local_array = atoi(argv[1]);
    int rank, number_processes;
    float local_sum = 0;
    float global_sum = 0;

    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&number_processes);

    // Create a random array of elements on all processes.
    srand(time(NULL)*rank);
    float *random_numbers = NULL;
    random_numbers = create_random_numbers(num_elems_local_array);

    // Sum the numbers locally
    for (int i = 0; i < num_elems_local_array; ++i) {
        local_sum += random_numbers[i];
    }
    // Print local sum and average on each process
    printf("Local sum for process %d = %f, avg = %f\n",rank,local_sum,
           local_sum/num_elems_local_array);

    // Reduce all of the local sums into the global sum
    MPI_Reduce(&local_sum,&global_sum,1,MPI_FLOAT,MPI_SUM,0,
               MPI_COMM_WORLD);

    // Print Global Results
    if (rank == 0) {
        printf("Total sum = %f, avg = %f\n",
               global_sum,global_sum/(num_elems_local_array*number_processes));
    }

    // Clean Up
    free(random_numbers);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();


}





