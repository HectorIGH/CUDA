/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 pi_buffon_needles.cu -o pibn
* and profile with nvprof --unified-memory-profiling off ./pibn
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

// Thread block size
#define THROWS_PER_THREAD 1024
#define BLOCKS 1024
#define THREADS 1024
#define PI 3.1415926535 //known value of PI
#define M_PI 3.14159265358979323846

__global__ void gpu_needles(float *estimate, curandState *states) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int hits = 0;
    float x, theta;

    curand_init(1234 * tid, tid, 0, &states[tid]); // Initialize CURAND

    for(int i = 0; i < THROWS_PER_THREAD; i++) {
        x = abs(curand_uniform(&states[tid])) ; // Mapping from [-1, 1] to [0, 1]
        theta = abs(curand_uniform(&states[tid])) * 0.5 * M_PI; // Mapping from [-1, 1] to [0, PI / 2]
        hits += (x <= 0.5 * sin(theta)); // Count if the crossing condition is satisfied
    }

    if (hits) {
        estimate[tid] = THROWS_PER_THREAD  / (float) hits; // Threads estimates of PI
    } else {
        estimate[tid] = 0.0;
    }
}

int main( void ) {

    float *host_estimate;
    float *device_estimate;
    curandState *devStates;

    // Allocate host memory
    host_estimate = (float*)malloc(sizeof(float) * BLOCKS * THREADS);

    // Allocate device memory 
    HANDLE_ERROR( cudaMalloc(&device_estimate, sizeof(float) * BLOCKS * THREADS) );
    HANDLE_ERROR( cudaMalloc(&devStates, sizeof(curandState) * BLOCKS * THREADS) );

    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // Executing kernel
    gpu_needles<<<BLOCKS, THREADS>>>(device_estimate, devStates);

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );

    // Copying data from device to host
    HANDLE_ERROR( cudaMemcpy(host_estimate, device_estimate, sizeof(float) * BLOCKS * THREADS, cudaMemcpyDeviceToHost) );

    // Freeing resources
    HANDLE_ERROR( cudaFree(device_estimate) );
    HANDLE_ERROR( cudaFree(devStates) );

    float PI_BY_GPU = 0;
    for(int i = 0; i < BLOCKS * THREADS; i++) {
        PI_BY_GPU += host_estimate[i];
    }
    PI_BY_GPU /= (BLOCKS * THREADS);

    printf( "\nTime taken:  %3.10f ms to calculate PI as %1.10f with an error of %1.10f.\n", elapsedTime, PI_BY_GPU, (PI_BY_GPU - PI) / PI);
}