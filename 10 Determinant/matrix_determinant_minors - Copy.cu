/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 matrix_determinant_minors.cu -o min
* and profile with nvprof --unified-memory-profiling off ./min
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

// Thread block size
#define N 3

typedef struct {
    int width;
    int height;
    double* elements;
} Matrix;


__global__ void determinant_by_minors(Matrix matrix, double *determinant, int n, int factorial, int depth) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int factor = 1;
    int j = 0;

    int new_dimension = n - 1;

    double prueba = 1;

    Matrix aux_matrix;
    aux_matrix.width = new_dimension;
    aux_matrix.height = new_dimension;

    cudaMalloc(&aux_matrix.elements, sizeof(double) * aux_matrix.width * aux_matrix.height);

    //for(int i = 0; i < matrix.height; i++){
    //    printf("|# ");
    //    for(int j = 0; j < matrix.width; j++) {
    //        printf("%.0f | ", matrix.elements[j + i * matrix.width]);
    //    }
    //    printf("**\n");
    //}
    
    if(n == 1) {
        //printf("\nScalar determinant %1.0f. index %d and compound index %d\n", matrix.elements[0], father, (index + father));
        prueba *= matrix.elements[0];
    } else {
        //printf("\nCalculate the minor for position %d with len %d\n", index, new_dimension);
        //printf("\nMinor Matrix for position %d with len %d\n", index, new_dimension);
        for(int i = n; i < n * n; i++) {
            if (i == index + n * factor) {
                factor++;
                continue;
            } else {
                aux_matrix.elements[j] = matrix.elements[i];
                //printf("\nFor minor %d extract the element %1.0f\n", index, matrix.elements[i]);
                prueba *= matrix.elements[i];
                j++;
            }
        }
        //printf("Prueba: %1.0f \n", prueba);
        determinant_by_minors<<<1, n - 1>>>(aux_matrix, determinant, n - 1, factorial, father + 1);
    }
    //printf("\nMatrix element %1.0f. and index %d\n", matrix.elements[index], index);

    //cudaFree(aux_matrix.elements);
}

int main( void ) {

    Matrix host_matrix;
    host_matrix.width = N;
    host_matrix.height = N;
    double *host_determinant;

    Matrix device_matrix;
    device_matrix.width = host_matrix.width;
    device_matrix.height = host_matrix.height;
    double *device_determinant;

    int factorial = 1;
    for(int i = 1; i <= N; i++) {
        factorial *= i;
    }


    // Allocate host memory
    host_matrix.elements = (double*)malloc(sizeof(double) * host_matrix.width * host_matrix.height);
    host_determinant = (double*)malloc(sizeof(double) * factorial * N);

    for(int i = 0;i < factorial * N; i++) {
        host_determinant[i] = 1000000;
    }

    // Initialize host Matrix
    for(int i = 0; i < host_matrix.width * host_matrix.height; i++){
        host_matrix.elements[i] = rand() % 100 + 1;
    }

    printf("\nOriginal Matrix:\n");
    for(int i = 0; i < host_matrix.height; i++){
        printf("| ");
        for(int j = 0; j < host_matrix.width; j++) {
            printf("%.0f | ", host_matrix.elements[j + i * host_matrix.width]);
        }
        printf("\n");
    }


    // Allocate device memory
    HANDLE_ERROR( cudaMalloc(&device_matrix.elements, sizeof(double) * device_matrix.width * device_matrix.height) );
    HANDLE_ERROR( cudaMalloc(&device_determinant, sizeof(double) * factorial * N) );

    // Transfer data from host to device memory
    HANDLE_ERROR( cudaMemcpy(device_determinant, host_determinant, sizeof(double)  * factorial * N, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(device_matrix.elements, host_matrix.elements, sizeof(double) * host_matrix.width * host_matrix.height, cudaMemcpyHostToDevice) );

    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // Executing kernel

    determinant_by_minors<<<1, N>>>(device_matrix, device_determinant, N, factorial, 0);

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "\nTime taken:  %3.10f ms\n", elapsedTime );

    // Transfer data from device to host memory
    HANDLE_ERROR( cudaMemcpy(host_determinant, device_determinant, sizeof(double) * factorial * N, cudaMemcpyDeviceToHost) );

    // Free resources
    HANDLE_ERROR( cudaFree(device_matrix.elements) );
    HANDLE_ERROR( cudaFree(device_determinant) );

    
    for(int i = 0; i < factorial * N; i++) {
        printf("\n Determinant: %1.0f\n", host_determinant[i]);
    }
}