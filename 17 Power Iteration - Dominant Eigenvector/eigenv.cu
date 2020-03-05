/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 eigenv.cu -o ev
* and profile with nvprof --unified-memory-profiling off ./ev
* check memory error with cuda_memcheck ./ev
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

extern const int N = 20;		//this defines the size of the matrix
extern const int ITERATIONS = 100;		//this defines the number of maximum iterations


__global__ void product_v(double *a, double *b, double *c) {

    int row = threadIdx.x;

    float element = 0;

    for(int i = 0; i < N; i++) {

        element += a[row * N + i] * b[i];

    }

    c[row] = element;
}


__global__ void norma(double *x, double *tam) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    double norm = 0;

    __shared__ double squares[N];

    squares[tid] = x[tid] * x[tid];


    __syncthreads();

    if(tid == 0) {
        for(int i = 0; i < N; i++) {
            norm += squares[i];
        }
        tam[0] = sqrt(norm);
        
        //for(int i = 0; i < N; i++) {      // Better if the matrix is small
        //    x[i] = x[i] / sqrt(norm);
        //}
    }
}

__global__ void normalize(double *x, double *tam) {  // Better if the matrix is big
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    x[tid] /= tam[0];
}

int main( int argc, char *argv[] ) {
    // Input in row order

    //double host_a[N * N] = {2, 1, 5, 7};
    // Normalized solution {0.17888544, 0.98386991} for eigenvector {1.34164079, 7.78151656} with eigenvalue 7.5

    //double host_a[N * N] = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    // Normalized solution {-0.33210408,  0.93704541,  0.107948} 
    // for eigenvector {-51.34277308, 147.15349546,  19.39163815} with eigenvalue 154.59

    //double host_a[N * N] = {10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -1, 0, 3, -1, 8};
    // Normalized solution {0.62120761, 0.46328147, 0.22845626, 0.58930393}
    // for eigenvector {7.97361894, 6.01434406, 2.47439245, 7.73944243} with eigenvalue 12.83

    double host_a[N * N]; // For random fill
    double host_x[N];
    double host_v[N];

    //srand((unsigned) time(NULL));
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            host_a[i * N + j] = rand() % 9 + 1; // For random fill
        }
        host_x[i] = 1.0;
    }

    double *dev_a;
	double *dev_x;
    double *dev_v;
    double *tam;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, (N * N) * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_x, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_v, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&tam, 1 * sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(dev_a, host_a, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, N * sizeof(double), cudaMemcpyHostToDevice));


    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    
    for(int i = 0; i < ITERATIONS; i++) {

        product_v<<<1, N>>>(dev_a, dev_x, dev_x);
        norma<<<1, N>>>(dev_x, tam);
        normalize<<<1, N>>>(dev_x, tam); // Better if the matrix is big

    }

    // To find the eigenvalue
    product_v<<<1, N>>>(dev_a, dev_x, dev_v); // now dev_v holds the eigenvector unormalized
	
    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "\nTime taken:  %3.10f ms\n", elapsedTime );


    HANDLE_ERROR(cudaMemcpy(host_x, dev_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_v, dev_v, N * sizeof(double), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_v));
    HANDLE_ERROR(cudaFree(tam));

    printf("\nOriginal Matrix:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", host_a[i * N + j]);
        }
        printf("\n");
    }

    printf("\nLeading eigenvector:\n");
    for(int i = 0; i < N; i++) {
        printf("%f ", host_x[i]);
    }
    printf("\n");

    printf("For eigenvalue: %f \n", host_v[0] / host_x[0]);
}