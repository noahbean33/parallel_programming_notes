/*
 Program: Basic point-to-point communication
 Purpose: Demonstrates a simple send from rank 0 to rank 1 using MPI_Send/MPI_Recv
 Compile: mpicc -O2 -o communication basic_communication.c
 Run:     mpirun -np 2 ./communication
 Notes:
 - Rank 0 sends a null-terminated string to rank 1 with tag=81.
 - Rank 1 receives, queries received size via MPI_Get_count, and prints it.
 MPI APIs used:
 - MPI_Init, MPI_Comm_rank
 - MPI_Send, MPI_Recv
 - MPI_Get_count
 - MPI_Finalize
*/
// compile with mpicc -o communication communication.c
// run with mpirun -np 2 ./communication

#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {

    int rank;
    int tag = 81;
    char msg[20];
    int sent_result, received_result;
    MPI_Status status;
    strcpy(msg,"Gold Coast");

    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank == 0) {
        sent_result = MPI_Send(msg,strlen(msg)+1,MPI_CHAR,
        1,tag,MPI_COMM_WORLD);
        if (sent_result != MPI_SUCCESS) {
            fprintf(stderr,"MPI_Send failed with error code %d\n",sent_result);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        else {
            printf("Process 0 sent message - %s - to Process 1\n",msg);
        }
    }
    else if (rank == 1) {
        received_result = MPI_Recv(msg,strlen(msg)+1,MPI_CHAR,0,tag,MPI_COMM_WORLD,&status);
        if (received_result != MPI_SUCCESS) {
            fprintf(stderr,"MPI_Recv failed with error code %d\n",received_result);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        else {
        int received_size;
        MPI_Get_count(&status,MPI_CHAR,&received_size);
        printf("Process 1 received message - %s - with size %d from Process 0\n",msg,received_size);
        }
    }

    MPI_Finalize();



}

