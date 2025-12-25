/*
 Program: Share heterogeneous data using MPI_Pack/MPI_Unpack and MPI_Bcast
 Purpose: Rank 0 packs an int and a double into a buffer and broadcasts it;
          other ranks unpack and print the values.
 Compile: mpicc -O2 -o sharingData sharingData.c
 Run:     mpirun -np <P> ./sharingData
 Input:   On rank 0, enter: <int> <double>; loop ends when either is negative.
 MPI APIs used:
 - MPI_Init, MPI_Comm_rank, MPI_Finalize
 - MPI_Pack, MPI_Unpack
 - MPI_Bcast
*/
#include <stdio.h>
#include "mpi.h"

int main(int argc, char* argv[]) {

    int rank;
    int a;
    double b;
    char packbuff[100];
    int packedsize;
    int retrieve_pos;

    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    do {
        packedsize = 0;
        if (rank == 0) {
            scanf("%d %lf", &a, &b);
            MPI_Pack(&a,1,MPI_INT,packbuff,100,&packedsize,MPI_COMM_WORLD);
            MPI_Pack(&b,1,MPI_DOUBLE,packbuff,100,&packedsize,MPI_COMM_WORLD);
        }

        MPI_Bcast(&packedsize,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(packbuff,packedsize,MPI_PACKED,0,MPI_COMM_WORLD);

        if (rank != 0) {
            retrieve_pos = 0;
            MPI_Unpack(packbuff,packedsize,&retrieve_pos,&a,1,MPI_INT,MPI_COMM_WORLD);
            MPI_Unpack(packbuff,packedsize,&retrieve_pos,&b,1,MPI_DOUBLE,MPI_COMM_WORLD);
        }

        printf("Process %d got %d and %lf\n",rank,a,b);

    }
    while (a >= 0 && b >= 0);

    MPI_Finalize();
    return 0;

}
