/*
 Program: Demonstrate cancelling a non-blocking send
 Purpose: Rank 0 starts an MPI_Isend then attempts MPI_Cancel; rank 1 sleeps
          then tries to receive. Shows how to check cancellation with MPI_Test_cancelled.
 Compile: mpicc -O2 -o mpi_cancel mpi_cancel.c
 Run:     mpirun -np 2 ./mpi_cancel
 Notes:
 - MPI_Cancel may not succeed if the operation has already matched; always check
   the status with MPI_Test_cancelled after MPI_Wait.
 - Rank 1 uses sleep(1) to increase chance the cancel succeeds.
 MPI APIs used:
 - MPI_Isend, MPI_Cancel, MPI_Wait, MPI_Test_cancelled
 - MPI_Recv
 - MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_Abort, MPI_Finalize
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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

    const int data_size = 10;
    int data[data_size];
    MPI_Request request;
    MPI_Status status;

    if (rank == 0) {
    // Initialize data array
        for (int i = 0; i < data_size; ++i) {
            data[i] = i;
        }

        // Start a non-blocking send
        MPI_Isend(data,data_size,MPI_INT,1,0,MPI_COMM_WORLD,&request);

        // Decide to cancel the send
        MPI_Cancel(&request);
        // Wait for cancellation to complete if it is possible
        MPI_Wait(&request,&status);

        // Check if it was cancelled
        int flag;
        MPI_Test_cancelled(&status,&flag);

        if (flag) {
            printf("Send operation was successfully cancelled.\n");
        }
        else {
            printf("Send operation was not cancelled.\n");
        }

    }

    else if (rank == 1) {
        sleep(1);
        int recv_buff[data_size];
        MPI_Status recv_status;
        int recv_result = MPI_Recv(recv_buff,data_size,MPI_INT,0,0,MPI_COMM_WORLD,&recv_status);
        if (recv_result == MPI_SUCCESS) {
            printf("Process 1 received data.\n");
        }
        else {
            printf("Process 0 cancelled the data.\n");
        }
    }

    MPI_Finalize();


    return 0;


}
