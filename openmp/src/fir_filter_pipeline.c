/*
 * This program demonstrates a producer-consumer pipeline using OpenMP tasks with dependencies.
 * A producer task computes a FIR filter and a consumer task applies a correction function.
 * The 'depend' clause ensures that the consumer task for an index 'i' runs only after the producer for 'i' is complete.
 *
 * Note:
 * 1. This is a code snippet. The constants INPUT_SIZE, TAP1, and TAP2 must be defined.
 * 2. The 'correction()' function and the initialization of 'sample', 'coeff1', and 'coeff2' arrays are missing.
 * 3. The 'final' array is not initialized, which will lead to incorrect results. It should be initialized to zero.
 *
 * Compilation instructions (assuming other necessary files and definitions are provided):
 * gcc -o fir_filter_pipeline -fopenmp Problem3.c -DINPUT_SIZE=<value> -DTAP1=<value> -DTAP2=<value> other_sources.c
 */
#include <omp.h>
float sample[INPUT_SIZE+TAP1];
float coeff1[TAP1], coeff2[TAP2];
float data_out[INPUT_SIZE], final[INPUT_SIZE];
int main() {
		// Create a parallel region. A single thread will generate all the tasks.
	#pragma omp parallel
	#pragma omp single
	float sum;
	for (int i=0; i<INPUT_SIZE; i++) {
				// Producer task: Applies a FIR filter.
		// It has a dependency on 'data_out[i]', indicating it writes to this location.
		#pragma omp task private(sum) depend(out: data_out[i])
		// Producer: Finite Impulse Response (FIR) filter
		sum=0.0;
		for (int j=0; j<TAP1; j++)
			sum += sample[i+j] * coeff1[j];
		data_out[i] = sum;
				// Consumer task: Applies a correction function.
		// It has a dependency on 'data_out[i]', indicating it reads from this location.
		// This task will wait until the producer task for the same index 'i' has completed.
		#pragma omp task depend (in: data_out[i])
		// Consumer: apply correction function
		for (int j=0; j<TAP2; j++)
						// Note: 'final[i]' should be initialized to 0 before this loop.
			final[i] += correction(data_out[i], coeff2[j]);
		}
	return 0;
}
