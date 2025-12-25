/*
 Program: Parallel data transformation with Scatter/Gather
 Purpose: Rank 0 creates DATA_SIZE integers, data is scattered equally; each rank
          applies transform_data() locally; results are gathered to rank 0.
 Compile: mpicc -O2 -o transformation transformation.c
 Run:     mpirun -np <P> ./transformation
 Notes:
 - DATA_SIZE must be divisible by number of processes.
 - Example transform squares each element.
 MPI APIs used:
 - MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_Finalize
 - MPI_Scatter, MPI_Gather
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


#define DATA_SIZE 100

int transform_data(int x) {
    return x*x;
}


int main() {

    int id, P;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&P);

    int local_data[DATA_SIZE/P];

    int* original_data = NULL;
    int* gathered_data = NULL;

    if (id == 0) {
        // Generate the original data on the root process
        original_data = (int *)malloc(sizeof(int)*DATA_SIZE);
        for (int i = 0; i < DATA_SIZE; ++i) {
            original_data[i] = i + 1;
        }

        gathered_data = (int *)malloc(sizeof(int)*DATA_SIZE);
    }

    // Scatter the original data to all processes
    MPI_Scatter(original_data,DATA_SIZE/P,MPI_INT,local_data,
                DATA_SIZE/P,MPI_INT,0,MPI_COMM_WORLD);


    // Apply a local transformation to the data
    for (int i = 0; i < DATA_SIZE/P; ++i) {
        local_data[i] = transform_data(local_data[i]);
    }

    // Gather the transformed data back to the root process
    MPI_Gather(local_data,DATA_SIZE/P,MPI_INT,gathered_data,
               DATA_SIZE/P,MPI_INT,0,MPI_COMM_WORLD);

    // Print the results on the root process
    if (id == 0) {
        printf("Original Data: ");
        for (int i = 0; i < DATA_SIZE; ++i) {
            printf("%d ", original_data[i]);
        }
        printf("\n");
        printf("Transformed Data: ");
        for (int i = 0; i < DATA_SIZE; ++i) {
            printf("%d ", gathered_data[i]);
        }
        printf("\n");

      // Free allocated memory on the root process
        free(original_data);
        free(gathered_data);
    }

    MPI_Finalize();

}
