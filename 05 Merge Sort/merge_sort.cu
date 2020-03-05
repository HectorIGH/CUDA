/*
* Compile using following structure
*nvcc -rdc=true -arch compute_35 merge_sort.cu -o sort
* and profile with nvprof --unified-memory-profiling off ./sort
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

#define N 256// 16384 32768 65536 131072 262144

__global__ void merge(float *A, float *B, float *C, int sizes) {
    int end = sizes - 1;
    int elemenst_in_A = sizes / 2;
    int elemenst_in_B = sizes / 2;
    //printf("\nEn merge working with:\n");
    for(int i = 0; i < elemenst_in_A; i++){
        //printf("%.0f ", A[i]);
    }
    //printf("\tand\t");
    for(int i = 0; i < elemenst_in_B; i++){
        //printf("%.0f ", B[i]);
    }
    //printf("\nFor C will have %d elements\n", sizes);
    while((elemenst_in_A > 0) && (elemenst_in_B > 0)) {
        //printf("\nFirst mientras\n");
        if(A[0] > B[0]) {
            for(int i = 0;i < end; i++) {
                C[i] = C[i + 1];
            }
            C[end] = B[0];
            //position -= 1;
            // Shifting. Simulating deletion
            for(int i = 0; i < elemenst_in_B - 1; i++) {
                B[i] = B[i + 1];
            }
            elemenst_in_B -= 1;
        } else {
            for(int i = 0;i < end; i++) {
                C[i] = C[i + 1];
            }
            C[end] = A[0];
            //position -= 1;
            // Shifting. Simulating deletion
            for(int i = 0;i < elemenst_in_A - 1; i++) {
                A[i] = A[i + 1];
            }
            elemenst_in_A -= 1;
        }
    }
    while(elemenst_in_A > 0) {
        //printf("\nSegundo mientras\n");
        for(int i = 0;i < end; i++) {
            C[i] = C[i + 1];
        }
        C[end] = A[0];
        //position -= 1;
        // Shifting. Simulating deletion
        for(int i = 0;i < elemenst_in_A - 1; i++) {
            A[i] = A[i + 1];
        }
        elemenst_in_A -= 1;
    }
    while(elemenst_in_B > 0) {
        //printf("\nTercer while\n");
        for(int i = 0;i < end; i++) {
            C[i] = C[i + 1];
        }
        C[end] = B[0];
        //position -= 1;
        // Shifting. Simulating deletion
        for(int i = 0;i < elemenst_in_B - 1; i++) {
            B[i] = B[i + 1];
        }
        elemenst_in_B -= 1;
    }

    //printf("\nObtuve:\n");
    for(int i = 0; i < sizes; i++){
        //printf("%.0f ", C[i]);
    }
    //printf("\n----Out of merge----\n");
}

__global__ void merge_sort(float *L, float *R, int size) {
    //int tid = threadIdx.x;
	//int number_of_threads = blockDim.x;

    // Getting size of arrays
    const int tamano = size / 2;

    //printf("\nThread %d of %d: \n", tid, number_of_threads);

    if (size == 1) {
        //printf("\nNada que hacer\n");
        return ;
    } else {
        if (threadIdx.x == 0) {
            float *left_L;
            float *left_R;

            cudaMalloc((void**)&left_L, sizeof(float) * tamano);
            cudaMalloc((void**)&left_R, sizeof(float) * tamano);

            for(int i = 0; i < tamano; i++) {
                left_L[i] = L[i];
                left_R[i] = L[tamano + i];
            }
            //printf("\nSplitted arrays L from %d:\n", threadIdx.x);
            for(int i = 0; i < tamano; i++){
                //printf("%.0f ", left_L[i]);
            }

            //printf("\nSplitted arrays R from %d:\n", threadIdx.x);
            for(int i = 0; i < tamano; i++){
                //printf("%.0f ", left_R[i]);
            }
            merge_sort<<<1, 2>>>(left_L, left_R, tamano);
            merge<<<1, 1>>>(left_L, left_R, L, size);
        } else {
            // Splitting right part
            float *right_L;
            float *right_R;

            cudaMalloc((void**)&right_L, sizeof(float) * tamano);
            cudaMalloc((void**)&right_R, sizeof(float) * tamano);

            for(int i = 0; i < tamano; i++) {
                right_L[i] = R[i];
                right_R[i] = R[tamano + i];
            }
            //printf("\nSplitted arrays else L from %d:\n", threadIdx.x);
            for(int i = 0; i < tamano; i++){
                //printf("%.0f ", right_L[i]);
            }

            //printf("\nSplitted arrays else R from %d:\n", threadIdx.x);
            for(int i = 0; i < tamano; i++){
                //printf("%.0f ", right_R[i]);
            }
            merge_sort<<<1, 2>>>(right_L, right_R, tamano);
            //__syncthreads();
            merge<<<1, 1>>>(right_L, right_R, R, size);
        }
    }
}


int main( void ) {

    float *host_array;
    float *host_array_L;
    float *host_array_R;
    float *device_array;
    float *device_array_L;
    float *device_array_R;

    // Allocate host memory
    host_array = (float*)malloc(sizeof(float) * N);
    host_array_L = (float*)malloc(sizeof(float) * N);
    host_array_R = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        host_array[i] = rand() % 100 + 1;
    }
    
    //printf("Original array:\n");
    for(int i = 0; i < N; i++){
        //printf("%.0f, ", host_array[i]);
    }

    //Slicing host_array into host_array_L and host_array_R
    for(int i = 0; i < N / 2; i++) {
        host_array_L[i] = host_array[i];
        host_array_R[i] = host_array[N / 2 + i];
    }

    //printf("\nSplitted arrays L:\n");
    for(int i = 0; i < N / 2; i++){
        //printf("%.0f ", host_array_L[i]);
    }
    
    //printf("\nSplitted arrays R:\n");
    for(int i = 0; i < N / 2; i++){
        //printf("%.0f ", host_array_R[i]);
    }

    // Allocate device memory 
    cudaMalloc((void**)&device_array, sizeof(float) * N);
    cudaMalloc((void**)&device_array_L, sizeof(float) * N);
    cudaMalloc((void**)&device_array_R, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(device_array_L, host_array_L, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_array_R, host_array_R, sizeof(float) * N, cudaMemcpyHostToDevice);

    // capture the start time
    //cudaEvent_t     start, stop;
    //HANDLE_ERROR( cudaEventCreate( &start ) );
    //HANDLE_ERROR( cudaEventCreate( &stop ) );
    //HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // Executing kernel
    merge_sort<<<1, 2>>>(device_array_L, device_array_R, N / 2);
    merge<<<1, 1>>>(device_array_L, device_array_R, device_array, N);

    // get stop time, and display the timing results
    //HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    //HANDLE_ERROR( cudaEventSynchronize( stop ) );
    //float   elapsedTime;
    //HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    //printf( "Time to order an array of %d elements:  %3.1f ms\n", N, elapsedTime );

    cudaMemcpy(host_array_L, device_array_L, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_array_R, device_array_R, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_array, device_array, sizeof(float) * N, cudaMemcpyDeviceToHost);

    //printf("\nSorted array:\n");
    for(int i = 0; i < N / 2; i++){
        //printf("%.0f ", host_array_L[i]);
    }
    //printf("\tand\t");
    for(int i = 0; i < N / 2; i++){
        //printf("%.0f ", host_array_R[i]);
    }

    //printf("\nOrdered array:\n");
    for(int i = 0; i < N; i++){
        //printf("%.0f, ", host_array[i]);
    }
}