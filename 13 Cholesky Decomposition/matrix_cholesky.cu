/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 matrix_cholesky.cu -o ch
* and profile with nvprof --unified-memory-profiling off ./ch
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

extern const int N = 3;

__global__ void second_step(double *L, double *a) {
    int tid = threadIdx.x;
    L[tid + 1] = a[tid + 1] / L[0];
}

__global__ void diagonal(double *L, double *a) {
    int row = blockDim.x; // The row index coincides with the number of products
    int tid = threadIdx.x;
    int pointer = row * N; // Points to the first object in a row
    __shared__ double squared[N]; // Over estimated memory. C cries if it is not a constant.

    squared[tid] = powf(L[pointer + tid], 2.0);

    __syncthreads();
    if(tid == 0) {
        double sum = 0;
        for(int i = 0; i < row; i++) {
            sum += squared[i];
        }
        sum = sqrtf(a[row * N + row] - sum);
        L[row * N + row] = sum;
    }
}

__global__ void inner(double *L, double *a) {
    int row = blockIdx.x + 2; // Row 0 is already fill. Row 1 only has diagonal and column; filled in other steps.
    int col = threadIdx.x + 1;

    int tid = threadIdx.x;
    //int pointer_row = row * N + 1 + tid;

    __shared__ double product[N]; // Over estimated memory. C cries if it is not a constant.

    //printf("blockIDx.x %d blockDim.x %d blockDim.y %d threadIdx.x %d threadIdx.y %d\n", blockIdx.x, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);

    if(col < row) {
        //printf("I am going to calculate te component L_%d_%d\n", row, col);
        product[tid] = L[row * N + tid] * L[col * N + tid];
        //printf("Which is %f\n", product[tid]);
    }

    __syncthreads();

    if(tid == 0) {
        double sum = 0;
        for(int i = 0; i < N - row; i++) {
            sum += product[i];
        }
        sum = a[row * N + col] - sum;
        L[row * N + col] = sum / L[row * N - row];
    }
}

void create_identity(double *m) {
    for(int i = 0; i < N; i++) {
        m[N * i + i] = 1;
    }
}

__global__ void solver(double *a, double *b, double *x, double *hold, int bid) {

    int tid = threadIdx.x;

    int index = N - bid;

    int last = (N * N - 1);

    __syncthreads();
    //hold[tid] = x[index + tid] * a[last - N * bid - tid];
    hold[tid] = x[N - 1 - tid] * a[last - N * bid - tid];

    //printf("\nProduct of a with: %f x %f in index %d for thread %d results in %f\n", a[last - N * bid - tid], x[N - 1 - tid], N - 1 - tid, tid, hold[tid]);


    //printf("\nCoeficient: %f", hold[index]);
    if(tid == 0) {
        double sum = 0;
        for (int i = 0; i < bid; i++) {
            sum += hold[i];
        }
        //printf("\nSum is %f and b %f and substract %f\n", sum, b[N - 1 - bid], b[N - 1 - bid ] - sum);
        x[N - 1 - bid] = (b[index + tid - 1] - sum) / a[last - N * bid - bid];
        b[N - 1 - bid] = x[N - 1 - bid];
        //printf("\nFinally coeficient: %f", x[N - 1 - bid]);
    }
    __syncthreads();
}

__global__ void solver_forward(double *a, double *b, double *x, double *hold, int i) {
    int tid = threadIdx.x;
    int col = i;
    __syncthreads();
    hold[tid] = x[tid] * a[col * N + tid];

    if(tid == 0) {
        double sum = 0;
        for(int i = 0; i < N; i++) {
            sum += hold[i];
        }
        x[col] = (b[col] - sum) / a[col * N + col];
        b[col] = x[col];
    }
    __syncthreads();
}

__global__ void traspose(double *a) {
    double aux[N * N];
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            aux[i + N * j] = a[i * N + j];
        }
    }

    for(int i = 0; i < N * N; i++) {
        a[i] = aux[i];
    }
}

void host_traspose(double *a) {
    double aux[N * N];
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            aux[i + N * j] = a[i * N + j];
        }
    }

    for(int i = 0; i < N * N; i++) {
        a[i] = aux[i];
    }
}

__global__ void product(double *a, double *b, double *c) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float element = 0;
    for(int i = 0; i < N; ++i) {
        element += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = element;
}

__global__ void product_v(double *a, double *b, double *c) {

    int row = threadIdx.x;

    float element = 0;
    for(int i = 0; i < N; ++i) {
        element += a[row * N + i] * b[i];
    }
    c[row] = element;
}

////////////////////////////////-------------------------------- **** --------------------------------////////////////////////////////
////////////////////////////////-------------------------------- MAIN --------------------------------////////////////////////////////
////////////////////////////////-------------------------------- **** --------------------------------////////////////////////////////

int main( void ) {
    // Input in column wise order. Fixed input but can be done automatically but not way to make sure they are L.I.
    // It is easier to fill matrix L in column wise order

    //double a[N * N] = {3, 5, 5, 0};
    //double b[N] = {3, 3}; 

    double a[N * N] = {4, 4, 2, 4, 5, 3, 2, 3, 3};
    double b[N] = {2, 3, 5}; // Solution is {1, -2, 3}

    //double a[N * N] = {9, 2, 6, 5, 2, 4, 3, 8, 6, 3, 7, 6, 5, 8, 6, 0};
    //double b[N] = {1, 7, 2, 5};

    //double a[N * N] = {4, 9, 4, 4, 6, 9, 1, 7, 3, 4, 4, 7, 5, 6, 9, 4, 3, 6, 0, 2, 6, 4, 9, 2, 6};
    //double b[N] = {6, 2, 4, 4, 6};

    //double a[N * N] = {8, 6, 5, 5, 9, 6, 6, 0, 2, 9, 0, 7, 5, 2, 1, 5, 9, 3, 5, 9, 5, 4, 7, 3, 9, 0, 9, 7, 4, 8, 6, 7, 3, 3, 8, 9};
    //double b[N] = {6, 3, 9, 3, 5, 8};

    //double a[N * N] = {2, 9, 0, 4, 3, 6, 6, 9, 1, 7, 3, 7, 5, 8, 0, 7, 1, 3, 0, 0, 9, 4, 3, 3, 7, 6, 3, 6, 3, 7, 0, 6, 6, 2, 2, 6, 5, 0, 3, 2, 7, 9, 6, 8, 9, 6, 2, 9, 2};
    //double b[N] = {1, 5, 7, 1, 10, 8, 8};

    //double a[N * N] = {5, 8, 4, 3, 0, 9, 1, 9, 8, 4, 8, 7, 6, 4, 7, 4, 4, 8, 0, 1, 3, 3, 4, 4, 3, 7, 1, 9, 8, 5, 7, 7, 0, 6, 3, 8, 7, 2, 5, 5, 9, 4, 3, 5, 2, 6, 7, 4, 1, 7, 4, 7, 5, 7, 3, 1, 9, 4, 4, 7, 5, 4, 1, 2};
    //double b[N] = {8, 8, 5, 5, 5, 9, 7, 4};



    double *x;
    double *host_y;
    double *host_L;
    double *host_L_T;
    
    x = (double*)malloc(sizeof(double) * N);
    host_y = (double*)malloc(sizeof(double) * N);
    host_L = (double*)malloc(sizeof(double) * N * N);
    host_L_T = (double*)malloc(sizeof(double) * N * N);

    for(int i = 0; i < N *N; i++) {
        host_L[i] = 0;
    }

    double *dev_A;
    double *dev_L;
    double *dev_b;
    double *dev_x;
    double *dev_y;

    HANDLE_ERROR(cudaMalloc((void**)&dev_A, (N * N) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_x, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_y, N * sizeof(double)));

	HANDLE_ERROR(cudaMemcpy(dev_A, a, (N * N) * sizeof(double), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)&dev_L, (N * N) * sizeof(double)));

    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // First step. One execution. Fill first element.
    host_L[0] = sqrt(a[0]);
    
    HANDLE_ERROR(cudaMemcpy(dev_L, host_L, N * N * sizeof(double), cudaMemcpyHostToDevice));

    // Second step. Parallel, N - 1 executions. Launch N - 1 threads. Fills first column.
    second_step<<<1, N - 1>>>(dev_L, dev_A);

    traspose<<<1, 1>>>(dev_L); // Easier for next steps
    traspose<<<1, 1>>>(dev_A);

    diagonal<<<1, 1>>>(dev_L, dev_A);

    for(int col = 1; col < N - 1; col ++) {
        // Fourth step. Launch, N - 2 blocks with N - 2 threads. Some threads will be idle. Fills the inner portion.
        inner<<<1, col>>>(dev_L, dev_A);
        // Third step. Launch N - 1 blocks with N - 1 threads. Some threads will be idle. Fills the diagonal.
        diagonal<<<1, col + 1>>>(dev_L, dev_A);
    }

    HANDLE_ERROR(cudaMemcpy(host_L, dev_L, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaMemcpy(dev_y, dev_b, N * sizeof(double), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(host_y, dev_b, N * sizeof(double), cudaMemcpyDeviceToHost));
    host_y[0] = host_y[0]  / host_L[0];
    HANDLE_ERROR(cudaMemcpy(dev_y, host_y, N * sizeof(double), cudaMemcpyHostToDevice));

    for(int i = 1; i < N; i++) {
        double *dev_hold;
        HANDLE_ERROR(cudaMalloc((void**)&dev_hold, (i) * sizeof(double)));
        solver_forward<<<1, i>>>(dev_L, dev_b, dev_y, dev_hold, i);
    }

    HANDLE_ERROR(cudaMemcpy(host_y, dev_y, N * sizeof(double), cudaMemcpyDeviceToHost));

    
    
    traspose<<<1, 1>>>(dev_L); // Transpose for the backward solve

    HANDLE_ERROR(cudaMemcpy(host_L_T, dev_L, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    /*
    HANDLE_ERROR(cudaMemcpy(dev_x, dev_b, N * sizeof(double), cudaMemcpyDeviceToDevice));
    */
    
    // Finally solving for x. First we need to include the posibility that X_N is not 1
    host_y[N - 1] = host_y[N - 1] / host_L_T[N * N - 1]; // To include the possible sing in the reduced matrix
    HANDLE_ERROR(cudaMemcpy(dev_x, host_y, N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_y, host_y, N * sizeof(double), cudaMemcpyHostToDevice));

    for(int i = 1; i < N; i++) {
        double *dev_hold;
        HANDLE_ERROR(cudaMalloc((void**)&dev_hold, (i) * sizeof(double)));
        solver<<<1, i>>>(dev_L, dev_x, dev_y, dev_hold, i);
    }

    HANDLE_ERROR(cudaMemcpy(x, dev_x, N * sizeof(double), cudaMemcpyDeviceToHost));

    

    // Get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "\nTime taken:  %3.10f ms\n", elapsedTime );

    HANDLE_ERROR(cudaFree(dev_A));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_L));
    HANDLE_ERROR(cudaFree(dev_y));

    printf("\n Original Extended Matrix\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", a[i * N + j]);
        }
        printf("\t %f\n", b[i]);
    }

    printf("\n L Matrix\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", host_L[i * N + j]);
        }
        printf("\n");
    }

    //printf("\n Y Vector: \n");
    //for(int i = 0; i < N; i++) {
    //    printf("%f\n", host_y[i]);
    //}

    printf("\n X Vector: \n");
    for(int i = 0; i < N; i++) {
        printf("%f\n", x[i]);
    }
}