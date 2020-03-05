/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 matrix_product.cu -o mat
* and profile with nvprof --unified-memory-profiling off ./mat
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

// Thread block size
#define BLOCK_SIZE 2

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

__global__ void product(Matrix a, Matrix b, Matrix c) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float element = 0;
    for(int i = 0; i < a.width; ++i) {
        element += a.elements[row * a.width + i] * b.elements[i * b.width + col];
        //printf("\n%1.0f times %1.0f\n", a.elements[row * a.width + i], b.elements[i * b.width + col]);
        //printf("\nCalculated %1.0f with blockIdx.y = %d blockDim.y = %d threadIdx.y = %d blockIdx.x = %d blockDim.x = %d threadIdx.x = %d\n", element, blockIdx.y, blockDim.y, threadIdx.y, blockIdx.x, blockDim.x, threadIdx.x);
    }
    //printf("\nCalculated %1.0f with blockIdx.y = %d blockDim.y = %d threadIdx.y = %d blockIdx.x = %d blockDim.x = %d threadIdx.x = %d\n", element, blockIdx.y, blockDim.y, threadIdx.y, blockIdx.x, blockDim.x, threadIdx.x);
    c.elements[row * c.width + col] = element;
}

int main( void ) {

    Matrix host_matrix_A;
    host_matrix_A.width = BLOCK_SIZE * 2;
    host_matrix_A.height = BLOCK_SIZE * 4;
    Matrix host_matrix_B;
    host_matrix_B.width = BLOCK_SIZE * 3;
    host_matrix_B.height = BLOCK_SIZE * 2;
    Matrix host_matrix_C;
    host_matrix_C.width = host_matrix_B.width;
    host_matrix_C.height = host_matrix_A.height;

    Matrix device_matrix_A;
    device_matrix_A.width = host_matrix_A.width;
    device_matrix_A.height = host_matrix_A.height;
    Matrix device_matrix_B;
    device_matrix_B.width = host_matrix_B.width;
    device_matrix_B.height = host_matrix_B.height;
    Matrix device_matrix_C;
    device_matrix_C.width = host_matrix_C.width;
    device_matrix_C.height = host_matrix_C.height;

    // Allocate host memory
    host_matrix_A.elements = (float*)malloc(sizeof(float) * host_matrix_A.width * host_matrix_A.height);
    host_matrix_B.elements = (float*)malloc(sizeof(float) * host_matrix_B.width * host_matrix_B.height);
    host_matrix_C.elements = (float*)malloc(sizeof(float) * host_matrix_C.width * host_matrix_C.height);

    // Initialize host Matrix
    for(int i = 0; i < host_matrix_A.width * host_matrix_A.height; i++){
        host_matrix_A.elements[i] = rand() % 100 + 1;
    }
    for(int i = 0; i < host_matrix_B.width * host_matrix_B.height; i++){
        host_matrix_B.elements[i] = rand() % 100 + 1;
    }
    
    printf("\nOriginal Matrix A:\n");
    for(int i = 0; i < host_matrix_A.height; i++){
        printf("| ");
        for(int j = 0; j < host_matrix_A.width; j++) {
            printf("%.0f | ", host_matrix_A.elements[j + i * host_matrix_A.width]);
        }
        printf("\n");
    }
    printf("\nOriginal Matrix B:\n");
    for(int i = 0; i < host_matrix_B.height; i++){
        printf("| ");
        for(int j = 0; j < host_matrix_B.width; j++) {
            printf("%.0f | ", host_matrix_B.elements[j + i * host_matrix_B.width]);
        }
        printf("\n");
    }


    // Allocate device memory 
    HANDLE_ERROR( cudaMalloc(&device_matrix_A.elements, sizeof(float) * device_matrix_A.width * device_matrix_A.height) );
    HANDLE_ERROR( cudaMalloc(&device_matrix_B.elements, sizeof(float) * device_matrix_B.width * device_matrix_B.height) );
    HANDLE_ERROR( cudaMalloc(&device_matrix_C.elements, sizeof(float) * device_matrix_C.width * device_matrix_C.height) );

    // Transfer data from host to device memory
    HANDLE_ERROR( cudaMemcpy(device_matrix_A.elements, host_matrix_A.elements, sizeof(float) * host_matrix_A.width * host_matrix_A.height, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(device_matrix_B.elements, host_matrix_B.elements, sizeof(float) * host_matrix_B.width * host_matrix_B.height, cudaMemcpyHostToDevice) );
    //HANDLE_ERROR( cudaMemcpy(device_matrix_C.elements, host_matrix_C.elements, sizeof(float) * host_matrix_C.width * host_matrix_C.height, cudaMemcpyHostToDevice) );

    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // Executing kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(host_matrix_B.width / dimBlock.x, host_matrix_A.height / dimBlock.y);
    product<<<dimGrid, dimBlock>>>(device_matrix_A, device_matrix_B, device_matrix_C);

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "\nTime taken:  %3.10f ms\n", elapsedTime );

    //HANDLE_ERROR( cudaMemcpy(host_matrix_A.elements, device_matrix_A.elements, sizeof(float) * host_matrix_A.width * host_matrix_A.height, cudaMemcpyDeviceToHost) );
    //HANDLE_ERROR( cudaMemcpy(host_matrix_B.elements, device_matrix_B.elements, sizeof(float) * host_matrix_B.width * host_matrix_B.height, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(host_matrix_C.elements, device_matrix_C.elements, sizeof(float) * host_matrix_C.width * host_matrix_C.height, cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaFree(device_matrix_A.elements) );
    HANDLE_ERROR( cudaFree(device_matrix_B.elements) );
    HANDLE_ERROR( cudaFree(device_matrix_C.elements) );

    printf("\nMatrix C:\n");
    for(int i = 0; i < host_matrix_C.height; i++){
        printf("| ");
        for(int j = 0; j < host_matrix_C.width; j++) {
            printf("%.0f | ", host_matrix_C.elements[j + i * BLOCK_SIZE]);
        }
        printf("\n");
    }
}