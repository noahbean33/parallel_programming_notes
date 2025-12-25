/*
 Program: Probe message to determine incoming size
 Purpose: Rank 0 sends a variable number of integers; rank 1 uses MPI_Probe and
          MPI_Get_count to determine the size, allocates buffer, and receives.
 Compile: mpicc -O2 -o program_probe program.c
 Run:     mpirun -np 2 ./program_probe
 Notes:
 - Demonstrates MPI_Probe/MPI_Get_count pattern for dynamic message sizes.
 MPI APIs used:
 - MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_Finalize, MPI_Abort
 - MPI_Send, MPI_Probe, MPI_Get_count, MPI_Recv
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {

    MPI_Init(&argc,&argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (size < 2) {
        fprintf(stderr,"We must have at least 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    int number_amount;
    if (rank == 0) {
        // Seed the random number generator
        srand(time(NULL) + rank); // Unique seed per process
        const int MAX_NUMBERS = 100;
        int numbers[MAX_NUMBERS];
        // Send varying amounts of data
        number_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;
        MPI_Send(numbers,number_amount,MPI_INT,1,0,MPI_COMM_WORLD);
        printf("Process 0 sent %d numbers to process 1\n",number_amount);
    }

    else if (rank == 1) {
        MPI_Status status;
        // Probe for an incoming message from process zero
        MPI_Probe(0,0,MPI_COMM_WORLD,&status);
        // Check how many numbers were actually sent
        MPI_Get_count(&status,MPI_INT,&number_amount);
        int* numb_buffer = (int*)malloc(sizeof(int) * number_amount);

        // Now receive the message with the allocated buffer
        MPI_Recv(numb_buffer,number_amount,MPI_INT,0,0,MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        printf("Process 1 received %d numbers from process 0\n",number_amount);

        free(numb_buffer);
    }

    MPI_Finalize();

    return 0;
}
