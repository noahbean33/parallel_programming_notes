/*
 Program: Hello world across MPI processes
 Purpose: Minimal example printing each process rank and world size.
 Compile: mpicc -O2 -o program hello.c
 Run:     mpirun -np <P> ./program
 MPI APIs used:
 - MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_Finalize
*/
// Compile with mpicc -o program hello.c
// To run mpirun -np 4 ./program

#include <mpi.h>
#include <stdio.h>

int main (int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    printf("Hello, I am process %d of %d\n",rank,size);
    MPI_Finalize();
    return 0;
}
