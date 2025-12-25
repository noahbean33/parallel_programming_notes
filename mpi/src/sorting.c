/*
 Program: Distributed sort via Scatter/Gather and local bubble sort
 Purpose: Rank 0 generates an array, data is scattered; each rank bubble-sorts its
          chunk; rank 0 gathers and performs an in-place merge-like insertion.
 Compile: mpicc -O2 -o sorting sorting.c
 Run:     mpirun -np <P> ./sorting
 Notes:
 - ARRAY_SIZE must be divisible by the number of processes.
 - Simple insertion-based merge at rank 0 is O(n^2) and for demonstration.
 MPI APIs used:
 - MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_Finalize
 - MPI_Scatter, MPI_Gather
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 100

#define MASTER 0

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int array[ARRAY_SIZE];
    int nump, rank;

    MPI_Init(&argc, &argv); // *----------------------

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nump);

    if (rank == 0) {

    srand(12345); // Seed for reproducibility
    for (int i = 0; i < ARRAY_SIZE; i++)
        array[i] = rand() % 100;

    printf("Original array: ");
    for (int i = 0; i < ARRAY_SIZE; i++)
        printf("%d ", array[i]);
    printf("\n");

    }

    int *subarray = malloc(ARRAY_SIZE/nump * sizeof(int));
    MPI_Scatter(array,ARRAY_SIZE/nump,MPI_INT,subarray,ARRAY_SIZE/nump,
                MPI_INT,MASTER,MPI_COMM_WORLD);


    // Sort subarrays
    bubbleSort(subarray,ARRAY_SIZE/nump);

    // Gather sorted subarrays at MASTER

    MPI_Gather(subarray,ARRAY_SIZE/nump,MPI_INT,array,ARRAY_SIZE/nump,
               MPI_INT,MASTER,MPI_COMM_WORLD);

    free(subarray);

    // Merge sorted subarrays at MASTER

    if (rank == 0) {
        for (int i = 1; i < nump; ++i) {
            int merge_index = i * (ARRAY_SIZE/nump);
            for (int j = 0; j < ARRAY_SIZE/nump; ++j) {
                int temp = array[merge_index + j];
                int k = merge_index + j - 1;
                while (k >= 0 && array[k] > temp) {
                    array[k+1] = array[k];
                    --k;
                }
                array[k+1] = temp;
            }
        }
        printf("Sorted array: ");
        for (int i = 0; i < ARRAY_SIZE; i++)
            printf("%d ", array[i]);
        printf("\n");
        return 0;

    }

    MPI_Finalize();
    return 0;

}

