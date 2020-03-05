/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 matrix_gauss.cu -o ga
* and profile with nvprof --unified-memory-profiling off ./ga
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

extern const int N = 8;

void create_identity(double *m) {
    for(int i = 0; i < N * N; i++) {
        m[i] = 0;
    }
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

    //printf("\ndev_A after traspose\n");
    //    for(int i = 0; i < N; i++) {
    //        for(int j = 0; j < N; j++) {
    //            printf("%f ", a[i * N + j]);
    //        }
    //        printf("\n");
    //    }
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

int main( void ) {
    // Input in column wise order. Fixed input but can be done automatically but not way to make sure they are L.I.
    // It is easier to fill matrix M_i in column wise order

    //double a[N * N] = {4, 4, 2, 4, 5, 3, 2, 3, 3};
    //double magicA[N * N] = {4, 4, 2, 4, 5, 3, 2, 3, 3};
    //double reduced[N * N] = {4, 4, 2, 4, 5, 3, 2, 3, 3};
    //double b[N] = {2, 3, 5};

    //double a[N * N] = {1, 1, 3, 3, 1, 11, 1, -1, 5};
    //double magicA[N * N] = {1, 1, 3, 3, 1, 11, 1, -1, 5};
    //double reduced[N * N] = {1, 1, 3, 3, 1, 11, 1, -1, 5};
    //double b[N] = {9, 1, 35};

    //double a[N * N] = {2, -3, -2, 1, -1, 1, -1, 2, 2};
    //double magicA[N * N] = {2, -3, -2, 1, -1, 1, -1, 2, 2};
    //double reduced[N * N] = {2, -3, -2, 1, -1, 1, -1, 2, 2};
    //double b[N] = {8, -11, -3};

    double a[N * N];
    double magicA[N * N];
    double reduced[N * N];
    double b[N];
    double b_print[N];

    //srand((unsigned) time(NULL));

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            a[i * N + j] = rand() % 10;
            magicA[i * N + j] = a[i * N + j];
            reduced[i * N + j] = a[i * N + j];
        }
        b[i] = rand() % 10;
        b_print[i] = b[i];
    }

    double *x;
    
    x = (double*)malloc(sizeof(double) * N);

    double *dev_A;
    double *dev_M;
    double *dev_R;
    double *dev_b;
    double *dev_b_R;
    double *dev_x;

    HANDLE_ERROR(cudaMalloc((void**)&dev_A, (N * N) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b_R, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_x, N * sizeof(double)));

	HANDLE_ERROR(cudaMemcpy(dev_A, a, (N * N) * sizeof(double), cudaMemcpyHostToDevice));

    traspose<<<1, 1>>>(dev_A); // Since a is column wise

    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)&dev_M, (N * N) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_R, (N * N) * sizeof(double)));

    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // Fill M matrix column wise and then traspose :)
    for(int col = 0; col < N - 1; col++) {

        double M[N * N];


        create_identity(M);

        int magic = N * col + col;

        for(int j = magic + 1; j < N * (col + 1); j++) {
            M[j] = - magicA[j] / magicA[magic];
        }
        
        host_traspose(M);

        //printf("\nM Matrix\n");
        //for(int i = 0; i < N; i++) {
        //    for(int j = 0; j < N; j++) {
        //        printf("%f ", M[i * N + j]);
        //    }
        //    printf("\n");
        //}

        HANDLE_ERROR(cudaMemcpy(dev_M, M, (N * N) * sizeof(double), cudaMemcpyHostToDevice));

        dim3 dimBlock(N, N);
        dim3 dimGrid(1, 1);
        product<<<dimGrid, dimBlock>>>(dev_M, dev_A, dev_R);
        HANDLE_ERROR(cudaMemcpy(magicA, dev_R, (N * N) * sizeof(double), cudaMemcpyDeviceToHost));

        host_traspose(magicA); // Since dev_R is row wise wee need magicA in column wise form

        HANDLE_ERROR(cudaMemcpy(dev_A, dev_R, (N * N) * sizeof(double), cudaMemcpyDeviceToDevice));


        product_v<<<1, N>>>(dev_M, dev_b, dev_b_R);
        HANDLE_ERROR(cudaMemcpy(b, dev_b_R, N * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(dev_b, dev_b_R, N * sizeof(double), cudaMemcpyDeviceToDevice));

        //printf("\nCoeficients:\n");
        //for(int i = 0; i < N; i++) {
        //    printf("%f \n", b[i]);
        //}

        //printf("\nMagicA\n");
        //for(int i = 0; i < N; i++) {
        //    for(int j = 0; j < N; j++) {
        //        printf("%f ", magicA[i * N + j]);
        //    }
        //    printf("\n");
        //}
    }
    HANDLE_ERROR(cudaFree(dev_M));
    HANDLE_ERROR(cudaFree(dev_R));
	
	HANDLE_ERROR(cudaMemcpy(reduced, dev_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(dev_x, dev_b, N * sizeof(double), cudaMemcpyDeviceToDevice));

    HANDLE_ERROR(cudaMemcpy(x, dev_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    x[N - 1] = x[N - 1] * (reduced[N * N - 1] / abs(reduced[N * N - 1])); // To include the possible sing in the reduced matrix
    HANDLE_ERROR(cudaMemcpy(dev_x, x, N * sizeof(double), cudaMemcpyHostToDevice));

    for(int i = 1; i < N; i++) {
        double *dev_hold;
        HANDLE_ERROR(cudaMalloc((void**)&dev_hold, (i) * sizeof(double)));
        solver<<<1, i>>>(dev_A, dev_b, dev_x, dev_hold, i);
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
    HANDLE_ERROR(cudaFree(dev_b_R));
    HANDLE_ERROR(cudaFree(dev_x));

    host_traspose(a);
    printf("\nOriginal Extended Matrix:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", a[i + j * N]);
        }
        printf("\t %f\n", b_print[i]);
    }

    host_traspose(magicA);
    printf("\nReduced Matrix\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", magicA[i * N + j]);
        }
        printf("\n");
    }

    //printf("\nReduce b coeficients:\n");
    //for(int i = 0; i < N; i++) {
    //    printf("%f \n", b[i]);
    //}

    printf("\nSolutions:\n");
    for(int i = 0; i < N; i++) {
        printf("%f \n", x[i]);
    }
}