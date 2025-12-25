/*
 Program: Bubble sort demo (single-process C)
 Purpose: Generates ARRAY_SIZE random integers, prints original and sorted arrays
          after applying bubble sort.
 Compile: gcc -O2 -o input input.c
 Run:     ./input
 Notes:
 - Not an MPI example; uses standard C only.
 - Bubble sort is O(n^2) and used here purely for demonstration.
*/
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 100

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

int main() {
    int array[ARRAY_SIZE];
    srand(12345); // Seed for reproducibility
    for (int i = 0; i < ARRAY_SIZE; i++)
        array[i] = rand() % 100;

    printf("Original array: ");
    for (int i = 0; i < ARRAY_SIZE; i++)
        printf("%d ", array[i]);
    printf("\n");

    bubbleSort(array, ARRAY_SIZE);

    printf("Sorted array: ");
    for (int i = 0; i < ARRAY_SIZE; i++)
        printf("%d ", array[i]);
    printf("\n");
    return 0;
}
