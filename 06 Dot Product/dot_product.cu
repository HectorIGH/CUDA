/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 dot_product.cu -o dot
* and profile with nvprof --unified-memory-profiling off ./dot
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

#define N 5

__global__ void sum(float *a) {
    int tid = threadIdx.x;
	int number_of_threads = blockDim.x;

    //printf("\nThread %d of %d\n", tid, number_of_threads);
    while(number_of_threads > 0) {

        if(tid < number_of_threads / 2) {

            a[tid] += a[tid + number_of_threads / 2];
            __syncthreads();
        }

        number_of_threads = number_of_threads / 2;
    }
}

__global__ void product(float *a, float *b, float *c, float *sum) {

    c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

    __syncthreads();

    int tid = threadIdx.x;
	int number_of_threads = blockDim.x;

    //printf("\nThread %d of %d\n", tid, number_of_threads);
    while(number_of_threads > 0) {

        if(tid < number_of_threads / 2) {

            c[tid] += c[tid + number_of_threads / 2];
            __syncthreads();
        }

        number_of_threads = number_of_threads / 2;
    }
    *sum = c[0];
}

__global__ void productTemp(float *a, float *b, float *c, float *sum) {
    __shared__ float temp[N];

    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
    
    __syncthreads();

    if(threadIdx.x == 0) {
        float suma = 0;
        for(int i = 0; i < N; i++) {
            suma += temp[i];
        }
        *sum = suma;
    }
}


int main( void ) {

    float *host_array_A;
    float *host_array_B;
    float *host_array_C;
    float *host_res;

    float *device_array_A;
    float *device_array_B;
    float *device_array_C;
    float *device_res;

    // Allocate host memory
    host_array_A = (float*)malloc(sizeof(float) * N);
    host_array_B = (float*)malloc(sizeof(float) * N);
    host_array_C = (float*)malloc(sizeof(float) * N);
    host_res = (float*)malloc(sizeof(float));

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        host_array_A[i] = rand() % 100 + 1;
        host_array_B[i] = rand() % 100 + 1;
        host_array_C[i] = rand() % 100 + 1;
    }
    
    printf("\nOriginal array A:\n");
    for(int i = 0; i < N; i++){
        printf("%.0f, ", host_array_A[i]);
    }
    printf("\nOriginal array B:\n");
    for(int i = 0; i < N; i++){
        printf("%.0f, ", host_array_B[i]);
    }


    // Allocate device memory 
    HANDLE_ERROR( cudaMalloc((void**)&device_array_A, sizeof(float) * N) );
    HANDLE_ERROR( cudaMalloc((void**)&device_array_B, sizeof(float) * N) );
    HANDLE_ERROR( cudaMalloc((void**)&device_array_C, sizeof(float) * N) );
    HANDLE_ERROR( cudaMalloc((void**)&device_res, sizeof(float)) );

    // Transfer data from host to device memory
    HANDLE_ERROR( cudaMemcpy(device_array_A, host_array_A, sizeof(float) * N, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(device_array_B, host_array_B, sizeof(float) * N, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(device_array_C, host_array_C, sizeof(float) * N, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(device_res, host_res, sizeof(float), cudaMemcpyHostToDevice) );

    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // Executing kernel
    //product<<<1, N>>>(device_array_A, device_array_B, device_array_C, device_res);
    productTemp<<<1, N>>>(device_array_A, device_array_B, device_array_C, device_res);

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "Time taken:  %3.10f ms\n", elapsedTime );

    HANDLE_ERROR( cudaMemcpy(host_array_A, device_array_A, sizeof(float) * N, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(host_array_B, device_array_B, sizeof(float) * N, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(host_array_C, device_array_C, sizeof(float) * N, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(host_res, device_res, sizeof(float), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaFree(device_array_A) );
    HANDLE_ERROR( cudaFree(device_array_B) );
    HANDLE_ERROR( cudaFree(device_array_C) );
    HANDLE_ERROR( cudaFree(device_res) );

    printf("\nReduction sum for dot product with %d elements: %.1f \n", N, host_res[0]);
    //for(int i = 0; i < N; i++){
    //    printf("%.0f, ", host_array_C[i]);
    //}
}