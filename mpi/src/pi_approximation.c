/*
 Program: Numerical approximation of PI using midpoint rule
 Purpose: Distributes the integration of 4/(1+x^2) over [0,1] across processes,
          then reduces partial sums to rank 0.
 Compile: mpicc -O2 -o pi_approximation pi_approximation.c -lm
 Run:     mpirun -np <P> ./pi_approximation
 Input:   Rank 0 reads integer n (number of intervals) from stdin; broadcast to all ranks.
 MPI APIs used:
 - MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_Finalize
 - MPI_Bcast, MPI_Reduce
*/
#include "mpi.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char* argv[]) {

    int id,num_processes;
    int n, i;
    double PI25DT = 3.141592653589793238462643;
    double h, mypi, pi, sum, x;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&num_processes);
    if (id == 0) {
        printf("Enter the number of intervals n: ");
        scanf("%d",&n);
    }
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

    h = 1.0 / (double) n;
    sum = 0.0;
    for (i = id + 1; i <= n; i += num_processes) {
        x = h * ((double)i -0.5);
        sum += 4.0 / (1+x*x);
    }

    mypi = h*sum;
    MPI_Reduce(&mypi,&pi,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    if (id == 0) {
        printf("PI is approximately %.16f, Error is %.16f\n",pi,
               fabs(pi-PI25DT));
    }

    MPI_Finalize();


}


