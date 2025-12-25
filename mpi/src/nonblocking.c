/*
 Program: Non-blocking point-to-point send/recv with computation overlap
 Purpose: Rank 0 posts MPI_Isend then computes a sum; rank 1 posts MPI_Irecv,
          waits, and computes a sum.
 Compile: mpicc -O2 -o nonblocking nonblocking.c
 Run:     mpirun -np 2 ./nonblocking
 MPI APIs used:
 - MPI_Isend, MPI_Irecv, MPI_Wait
 - MPI_Init, MPI_Comm_rank, MPI_Finalize
*/
#include <stdio.h>
#include <mpi.h>

#define MESSAGE_SIZE 8

int main(int argc, char* argv[]) {
    int rank;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if (rank == 0) {
        MPI_Request send_request;
        // Initialize Array
        int message[MESSAGE_SIZE];
        for (int i = 0; i < MESSAGE_SIZE; ++i) {
            message[i] = i;
        }
        // Non-blocking send from Process 0 to Process 1
        MPI_Isend(message,MESSAGE_SIZE,MPI_INT,1,0,
                  MPI_COMM_WORLD,&send_request);

        // Process 0: Compute the sum of the elements
        int sum = 0;
        for (int i = 0; i < MESSAGE_SIZE; ++i) {
            sum += message[i];
        }

        printf("Process 0 computed the sum: %d\n",sum);

        // Wait for the completion of the send operation
        MPI_Wait(&send_request,MPI_STATUS_IGNORE);

        printf("Process 0 sent a message to Process 1\n");

    }
    else if (rank == 1) {
        // Process 1 posts a non-blocking receive
        int received_message[MESSAGE_SIZE];
        MPI_Request recv_request;
        // Non-blocking receive in process 1
        MPI_Irecv(received_message,MESSAGE_SIZE,MPI_INT,
                  0,0,MPI_COMM_WORLD,&recv_request);


        // Wait for the completion of the receive operation
        MPI_Wait(&recv_request,MPI_STATUS_IGNORE);
        printf("Process 1 received a message from Process 0\n");

        // Process 1: Compute the sum of the elements
        int sum = 0;
        for (int i = 0; i < MESSAGE_SIZE; ++i) {
            sum += received_message[i];
        }
        printf("Process 1 computed the sum: %d\n",sum);

    }

    MPI_Finalize();
    return 0;
}


