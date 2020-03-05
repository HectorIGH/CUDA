/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 matrix_determinant_permutations.cu -o det
* and profile with nvprof --unified-memory-profiling off ./det
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

// Thread block size
#define N 2

int permutation_index = 0; // I am really sorry for this :(

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


__global__ void determinant_by_permutations(Matrix matrix, int *permutations, float *sign, float *determinant) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int position = y * N + x;
    float current_product = sign[position];
    
    for(int i = 0; i < N; i++) {
        int origin = i * N;
        int sigma = permutations[position * N + i];
        current_product *= matrix.elements[origin + sigma];
        //printf("\nAnd current product inside for %1.0f\n", current_product);
        //printf("\nMatrix element for position %d is %1.0f\n", position, matrix.elements[origin + sigma]);
    }
    //printf("\nAnd current product after permutation %d is %1.0f\n", position, current_product);
    determinant[0] += current_product * sign[position];
}

__global__ void bubble_counter(int *Sn, int number_of_permutations, float *counter) {
    int tid = threadIdx.x;
    int index = tid * N;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if ((Sn[index + i] > Sn[index + j]) && (i < j)) {
                counter[tid] *= -1;
            }
        }
    }
}

void permutations(int *host_permutations_list, int *original_list, int start, int end) {
    if (start == end) {
        for (int j = 0; j < N; j++) {
            host_permutations_list[permutation_index] = original_list[j];
            permutation_index += 1;
        }
    } else {
        for (int i = start; i <= end; i++) {
            float temp = original_list[start];
            original_list[start] = original_list[i];
            original_list[i] = temp;
            host_permutations_list[start * N] = temp;
            permutations(host_permutations_list, original_list, start + 1, end);
            temp = original_list[start];
            original_list[start] = original_list[i];
            original_list[i] = temp;
        }
    }
}

int main( void ) {

    int number_of_permutations = 1;
    for (int i = 1; i <= N; i++) {
        number_of_permutations *= i;
    }

    int *host_original_list;
    int *host_permutations_list;
    float *host_sign;
    Matrix host_matrix;
    host_matrix.width = N;
    host_matrix.height = N;
    float *host_determinant;

    int *device_original_list;
    int *device_permutations_list;
    float *device_sign;
    Matrix device_matrix;
    device_matrix.width = host_matrix.width;
    device_matrix.height = host_matrix.height;
    float *device_determinant;


    // Allocate host memory
    host_original_list = (int*)malloc(sizeof(int) * N);
    host_permutations_list = (int*)malloc(sizeof(int) * number_of_permutations * N);
    host_sign = (float*)malloc(sizeof(float) * number_of_permutations);
    host_matrix.elements = (float*)malloc(sizeof(float) * host_matrix.width * host_matrix.height);
    host_determinant = (float*)malloc(sizeof(float) * 1);

    host_determinant[0] = 0;

    // Initialize host list for permutations
    for(int i = 0; i < N; i++){
        host_original_list[i] = i;
    }
    for(int i = 0; i < number_of_permutations; i++){
        host_sign[i] = 1;
    }

    // Initialize host Matrix
    for(int i = 0; i < host_matrix.width * host_matrix.height; i++){
        host_matrix.elements[i] = rand() % 100 + 1;
    }

    
    //printf("\nOriginal List:\n");
    //for(int i = 0; i < N; i++){
    //    printf("%d ", host_original_list[i]);
    //}
    //printf("\n");


    // Allocate device memory
    HANDLE_ERROR( cudaMalloc(&device_original_list, sizeof(int) * N) );
    HANDLE_ERROR( cudaMalloc(&device_permutations_list, sizeof(int) * number_of_permutations * N) );
    HANDLE_ERROR( cudaMalloc(&device_sign, sizeof(float) * number_of_permutations) );
    HANDLE_ERROR( cudaMalloc(&device_matrix.elements, sizeof(float) * device_matrix.width * device_matrix.height) );
    HANDLE_ERROR( cudaMalloc(&device_determinant, sizeof(float) * 1) );

    // Transfer data from host to device memory
    HANDLE_ERROR( cudaMemcpy(device_original_list, host_original_list, sizeof(int) * N, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(device_sign, host_sign, sizeof(float) * number_of_permutations, cudaMemcpyHostToDevice) );
    //HANDLE_ERROR( cudaMemcpy(device_determinant, host_determinant, sizeof(float)  *1, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(device_matrix.elements, host_matrix.elements, sizeof(float) * host_matrix.width * host_matrix.height, cudaMemcpyHostToDevice) );

    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // Executing kernel
    permutations(host_permutations_list, host_original_list, 0, N - 1);

    // Copying the permutations from host to device
    HANDLE_ERROR( cudaMemcpy(device_permutations_list, host_permutations_list, sizeof(int) * number_of_permutations * N, cudaMemcpyHostToDevice) );
    bubble_counter<<<1, number_of_permutations>>>(device_permutations_list, number_of_permutations, device_sign);

    int THREADS = 1;
    for(int i = 1; i < N; i++) {
        THREADS *= i;
    }
    determinant_by_permutations<<<N, THREADS>>>(device_matrix, device_permutations_list, device_sign, device_determinant);

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "\nTime taken:  %3.10f ms\n", elapsedTime );

    // Transfer data from device to host memory
    //HANDLE_ERROR( cudaMemcpy(host_permutations_list, device_permutations_list, sizeof(float) * number_of_permutations, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(host_sign, device_sign, sizeof(float) * number_of_permutations, cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(host_determinant, device_determinant, sizeof(float) * 1, cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaFree(device_original_list) );
    HANDLE_ERROR( cudaFree(device_permutations_list) );
    HANDLE_ERROR( cudaFree(device_sign) );
    HANDLE_ERROR( cudaFree(device_matrix.elements) );
    HANDLE_ERROR( cudaFree(device_determinant) );

    //printf("\nPermutations:\n");
    //for(int i = 0; i < number_of_permutations; i++){
        //printf("%d. ", i);
        //for(int j = 0; j < N; j++) {
            //printf("%d | ", host_permutations_list[i * N + j]);
        //}
        //printf("\n");
    //}

    //printf("\nNumber of inversions:\n");
    //for(int i = 0; i < number_of_permutations; i++){
        //printf("%1.0f \n", host_sign[i]);
    //}

    printf("\nOriginal Matrix:\n");
    for(int i = 0; i < host_matrix.height; i++){
        printf("| ");
        for(int j = 0; j < host_matrix.width; j++) {
            printf("%.0f | ", host_matrix.elements[j + i * host_matrix.width]);
        }
        printf("\n");
    }

    //printf("\n Determinant: %1.0f\n", host_determinant[0]);
}