/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 jacobi.cu -o jb
* and profile with nvprof --unified-memory-profiling off ./jb
* check memory error with cuda_memcheck ./jb
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

extern const int N = 8;		//this defines the size of the matrix
extern const int ITERATIONS = 10;		//this defines the number of maximum iterations


__global__ void product_v(double *a, double *b, double *c) {

    int row = threadIdx.x;

    float element = 0;

    for(int i = 0; i < N; i++) {

        element += a[row * N + i] * b[i];

    }

    c[row] = element;
}

__global__ void diagonal_inverse(double *matrix) {
    int tid = threadIdx.x;
    int index = N * tid + tid;
    matrix[index] = 1 / matrix[index];
}


__global__ void product(double *a, double *b, double *c) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float element = 0;
    for(int i = 0; i < N; ++i) {
        element += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = element;
    //printf("Prod: %f \t", c[row * N + col]);
}

__global__ void sum_special(double *a, double *b, double *c) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    int dim = blockDim.x;
    c[row * dim + col] = -1.0 * (a[row * dim + col] + b[row * dim + col]);
}

__global__ void sum(double *a, double *b, double *c) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    int dim = blockDim.x;

    c[row * dim + col] = a[row * dim + col] + b[row * dim + col];
}

__global__ void decompose(double *A, double *D, double *U, double *L) {
    int col = threadIdx.x;
    int row = threadIdx.y;
    int dim = blockDim.x;
    //printf("(%d, %d) in dim %d\n", row, col, dim);
    if(row == col) { // Diagonal
        D[row * dim + row] = A[row * dim + row];
    }
    if(row < col){ // Upper
        U[row * dim + col] = A[row * dim + col];
    }
    if(col < row){ // Lower
        L[row * dim + col] = A[row * dim + col];
    }
}


int main( void ) {
    // Input in row order
    //double host_a[N * N] = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    //double host_b[N] = {2, 3, 5}; // Solution is {0.028163, -0.044408, -0.15069}

    //double host_a[N * N] = {2, 1, 5, 7};
    //double host_b[N] = {11, 13}; // Solution is {7.111, -3.222}

    //double host_a[N * N] = {10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -1, 0, 3, -1, 8};
    //double host_b[N] = {6, 25, -11, 15}; // Solution is {1, 2, -1, 1}

    double host_a[N * N]; // For random fill
    double host_b[N]; // For random fill

    double host_d[N * N];
    double host_u[N * N];
    double host_l[N * N];
    double host_t[N * N];

    double host_c[N];

    double host_x[N];
    
    //srand((unsigned) time(NULL));
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            host_a[i * N + j] = rand() % 3 + 1; // For random fill
            host_d[i * N + j] = 0.0;
            host_u[i * N + j] = 0.0;
            host_l[i * N + j] = 0.0;
        }
        host_b[i] = rand() % 1000 + 1; // For random fill
        host_x[i] = 1.0;
    }

    double *dev_a;
    double *dev_d;
    double *dev_u;
    double *dev_l;
    double *dev_t;
    double *dev_c;
	double *dev_b;
	double *dev_x;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, (N * N) * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_d, (N * N) * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_u, (N * N) * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_l, (N * N) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_t, (N * N) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_x, N * sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(dev_a, host_a, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_d, host_d, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_u, host_u, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_l, host_l, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, host_b, N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, N * sizeof(double), cudaMemcpyHostToDevice));


    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    dim3 THREADS(N, N);
    decompose<<<1, THREADS>>>(dev_a, dev_d, dev_u, dev_l);

    diagonal_inverse<<<1, N>>>(dev_d);

    sum_special<<<1, THREADS>>>(dev_l, dev_u, dev_t);

    product<<<1, THREADS>>>(dev_d, dev_t, dev_t);

    product_v<<<1, N>>>(dev_d, dev_b, dev_c);

    for(int i = 0; i < ITERATIONS; i++) {
        product_v<<<1, N>>>(dev_t, dev_x, dev_x);
        sum<<<1, N>>>(dev_x, dev_c, dev_x);
    }
	
    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "\nTime taken:  %3.10f ms\n", elapsedTime );


    HANDLE_ERROR(cudaMemcpy(host_x, dev_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_d, dev_d, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_u, dev_u, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_l, dev_l, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_t, dev_t, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_c, dev_c, N * sizeof(double), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_d));
    HANDLE_ERROR(cudaFree(dev_u));
    HANDLE_ERROR(cudaFree(dev_l));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_x));

    printf("\nOriginal Extended Matrix:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", host_a[i * N + j]);
        }
        printf("\t %f\n", host_b[i]);
    }

    //printf("\nD Matrix:\n");
    //for(int i = 0; i < N; i++) {
    //    for(int j = 0; j < N; j++) {
    //        printf("%f ", host_d[i * N + j]);
    //    }
    //    printf("\n");
    //}

    //printf("\nU Matrix:\n");
    //for(int i = 0; i < N; i++) {
    //    for(int j = 0; j < N; j++) {
    //        printf("%f ", host_u[i * N + j]);
    //    }
    //    printf("\n");
    //}

    //printf("\nL Matrix:\n");
    //for(int i = 0; i < N; i++) {
    //    for(int j = 0; j < N; j++) {
    //        printf("%f ", host_l[i * N + j]);
    //    }
    //    printf("\n");
    //}

    //printf("\nT Matrix:\n");
    //for(int i = 0; i < N; i++) {
    //    for(int j = 0; j < N; j++) {
    //        printf("%f ", host_t[i * N + j]);
    //    }
    //    printf("\n");
    //}

    //printf("\nC vector:\n");
    //for(int i = 0; i < N; i++) {
    //    printf("%f ", host_c[i]);
    //}
    //printf("\n");

    printf("\nX Solutions:\n");
    for(int i = 0; i < N; i++) {
        printf("%f ", host_x[i]);
    }
    printf("\n");
}