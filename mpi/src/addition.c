/*
 Program: Parallel vector addition using MPI_Scatter and MPI_Gather
 Purpose: Demonstrates distributing two integer vectors across processes, computing
          element-wise addition locally, and gathering results on the master.
 Compile: mpicc -O2 -o addition addition.c
 Run:     mpirun -np <P> ./addition
 Notes:
 - Problem size is PROBLEM_SIZE (must be divisible by number of processes).
 - Rank 0 initializes vectors A and B with random values.
 - Each process receives n/P elements to add; result is gathered to rank 0.
 MPI APIs used:
 - MPI_Init, MPI_Comm_size, MPI_Comm_rank
 - MPI_Bcast (broadcast problem size)
 - MPI_Scatter (distribute slices of A and B)
 - MPI_Gather (collect slices of C)
 - MPI_Finalize
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MASTER 0
#define PROBLEM_SIZE 12


int main(int argc, char *argv[]) {

    int *A, *B, *C;
    int total_processes;
    int rank;

    int n_per_process;
    int n;

    // Initialize MPI environment
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&total_processes);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    // Local arrays
    int *ap,*bp,*cp;


    if (rank == MASTER) {
        n = PROBLEM_SIZE;

        A = (int *) malloc(sizeof(int)*n);
        B = (int *) malloc(sizeof(int)*n);
        C = (int *) malloc(sizeof(int)*n);

        // Initialize A and B
        srand(time(NULL));
        for (int i = 0; i < n; ++i) {
            A[i] = (rand() % 10) + 1;
            B[i] = (rand() % 10) + 1;
        }
    }

    /* Distribute problem size in variable n from master to workers */

    MPI_Bcast(&n,1,MPI_INT,MASTER,MPI_COMM_WORLD);

    n_per_process = n/total_processes;
    /* Allocate space required for local arrays */
    ap = (int *) malloc(sizeof(int)*n_per_process);
    bp = (int *) malloc(sizeof(int)*n_per_process);
    cp = (int *) malloc(sizeof(int)*n_per_process);

    /* Distribute parts of vectors A and B from master to workers */
    MPI_Scatter(A,n_per_process,MPI_INT,ap,n_per_process,
                MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(B,n_per_process,MPI_INT,bp,n_per_process,
                MPI_INT,0,MPI_COMM_WORLD);

    for (int i = 0; i < n_per_process; ++i) {
        cp[i] = ap[i] + bp[i];
    }

    /* Gather parts of result vector c in the master*/
    MPI_Gather(cp,n_per_process,MPI_INT,C,n_per_process,
               MPI_INT,MASTER,MPI_COMM_WORLD);

    // Print all the information

    if (rank == 0) {

        printf("Vector A:\n");
        for (int i = 0; i < PROBLEM_SIZE; ++i) {
            printf("%d ",A[i]);
        }

        printf("\n");

        printf("Vector B:\n");
        for (int i = 0; i < PROBLEM_SIZE; ++i) {
            printf("%d ",B[i]);
        }

        printf("\n");

        printf("Vector C:\n");
        for (int i = 0; i < PROBLEM_SIZE; ++i) {
            printf("%d ",C[i]);
        }

        printf("\n");

        free(A);
        free(B);
        free(C);
    }

    free(ap);
    free(bp);
    free(cp);

    // Finalize MPI environment
    MPI_Finalize();


    return 0;

}




