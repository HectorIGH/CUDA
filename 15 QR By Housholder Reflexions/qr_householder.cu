/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 qr_householder.cu -o hh
* and profile with nvprof --unified-memory-profiling off ./hh
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

extern const int N = 8;		//this defines the number of elements in each vector

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

__global__ void product_v(double *a, double *b, double *c) {

    int row = threadIdx.x;

    float element = 0;
    for(int i = 0; i < N; ++i) {
        element += a[row * N + i] * b[i];
    }
    c[row] = element;
}

__global__ void diagonal_inverse(double *matrix) {
    int tid = threadIdx.x;
    int index = N * tid + tid;
    matrix[index] = 1 / matrix[index];
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


__global__ void product(double *a, double *b, double *c) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float element = 0;
    for(int i = 0; i < N; ++i) {
        element += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = element;
}


__global__ void householder(double *A, double *Qt, double *R, double *G, int col) {
    //int col = threadIdx.x;

    __syncthreads();
    double *x;
    double *e;
    double *u;
    double *v;
    cudaMalloc((void**)&x, (N - col) * sizeof(double));
    cudaMalloc((void**)&e, (N - col) * sizeof(double));
    cudaMalloc((void**)&u, (N - col) * sizeof(double));
    cudaMalloc((void**)&v, (N - col) * sizeof(double));

    for(int i = 0; i < N - col; i++) {
        x[i] = R[col * N + col + i * N];
        e[i] = 0;
        //printf("%d-%f ", col, x[i]);
    }
    //printf("\n");
    double norma = 0;
    for(int i = 0; i < N - col; i++) {
        norma += pow(x[i], 2.0);
    }
    e[0] = -1.0 * sqrt(norma) * A[col * N + col] / abs(A[col * N + col]);
    for(int i = 0; i < N - col; i++) {
        //printf("e(%d-%f) ", col, e[i]);
    }
    for(int i = 0; i < N - col; i++) {
        u[i] = x[i] + e[i];
        //printf("u(%d-%f) ", col, u[i]);
    }
    norma = 0;
    for(int i = 0; i < N - col; i++) {
        norma += pow(u[i], 2.0);
    }
    norma = sqrt(norma);
    for(int i = 0; i < N - col; i++) {
        v[i] = u[i] / norma;
        //printf("v(%d-%f) ", col, v[i]);
    }

    // Creating outer product
    int dim = N - col;
    double aux[N * N];
    //printf("Outer\n");
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            aux[i * dim + j] = v[i] * v [j];
            //printf("(%d)%f ", col, aux[i * dim + j]);
        }
        //printf("\n");
    }

    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            G[i * N + j + (col * N + col)] -= 2.0 * aux[i * dim + j];
        }
    }

    //printf("\nQ\n");
    //for(int i = 0; i < N; i++) {
        //for(int j = 0; j < N; j++) {
            //printf("(%d)%f ", col, G[i * N + j]);
        //}
        //printf("\n");
    //}

    __syncthreads();
    dim3 dimBlockG(N, N);
    dim3 dimGridG(1, 1);
    product<<<dimGridG, dimBlockG>>>(G, R, R);

    __syncthreads();
    

    //printf("R\n");
    //for(int i = 0; i < N; i++) {
        //for(int j = 0; j < N; j++) {
            //printf("(%d)%f ", col, R[i * N + j]);
        //}
        //printf("\n");
    //}

    product<<<dimGridG, dimBlockG>>>(Qt, G, Qt);
    
        
    
}


int main( void ) {
    // Input in row order
    //double input[N * N] = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    //double a[N * N] = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    //double r[N * N] = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    //double b[N] = {2, 3, 5};

    //double input[N * N] = {4, 4, 2, 4, 5, 3, 2, 3, 3};
    //double a[N * N] = {4, 4, 2, 4, 5, 3, 2, 3, 3};
    //double r[N * N] = {4, 4, 2, 4, 5, 3, 2, 3, 3};
    //double b[N] = {2, 3, 5};

    
    double input[N * N];
    double a[N * N];
    double r[N * N];
    double b[N];
    

    //srand((unsigned) time(NULL));
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            input[i * N + j] = rand() % 9 + 1;
            a[i * N + j] = input[i * N + j];
            r[i * N + j] = input[i * N + j];
        }
        b[i] = rand() % 10 + 1;
    }


    double *host_d;
    double *host_y;
    double *host_x;

    host_d = (double*)malloc(sizeof(double) * N * N);
    host_y = (double*)malloc(sizeof(double) * N);
    host_x = (double*)malloc(sizeof(double) * N);

    double host_eye[N * N];
    double host_G[N * N];

    for(int i = 0; i < N * N; i++) {
        host_eye[i] = 0;
        host_G[i] = 0;
    }
    for(int i = 0; i < N; i++) {
        host_eye[N * i + i] = 1;
        host_G[N * i + i] = 1;
    }

    double *dev_R;
    double *dev_Q_t;
    double *dev_D;
    double *dev_A;
	double *dev_input;
	double *dev_m;
    double *dev_y;
    double *dev_aux;
    double *dev_b;
    double *dev_x;
    double *G;

	HANDLE_ERROR(cudaMalloc((void**)&dev_input, (N * N) * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_m, sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_R, (N * N) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_Q_t, (N * N) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_D, (N * N) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_A, (N * N) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_aux, (N * N) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_y, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_x, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&G, (N * N) * sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(dev_input, input, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_R, input, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_Q_t, host_eye, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(G, host_G, (N * N) * sizeof(double), cudaMemcpyHostToDevice));

    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    traspose<<<1, 1>>>(dev_input);
    for(int i = 0; i < N - 1; i++) {
        HANDLE_ERROR(cudaMemcpy(G, host_G, (N * N) * sizeof(double), cudaMemcpyHostToDevice));
        householder<<<1, 1>>>(dev_input, dev_Q_t, dev_R, G, i);
    }

	
    // Solve linear equations
	
	HANDLE_ERROR(cudaMemcpy(input, dev_Q_t, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    // Here input and dev_input holds the transpose of Q. Q^T

    HANDLE_ERROR( cudaMemcpy(r, dev_R, sizeof(double) * N * N, cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(dev_input, dev_Q_t, sizeof(double) * N * N, cudaMemcpyDeviceToDevice));
    
    // Calculating Q_t * Q = D
    traspose<<<1, 1>>>(dev_Q_t); // Not necessary since dev_input and hence dev_Q_t already holds the transpose.
    //traspose<<<1, 1>>>(dev_input); // We get the real Q matrix
    dim3 dimBlock(N, N);
    dim3 dimGrid(1, 1);
    product<<<dimGrid, dimBlock>>>(dev_Q_t, dev_input, dev_D);

    HANDLE_ERROR( cudaMemcpy(host_d, dev_D, sizeof(double) * N * N, cudaMemcpyDeviceToHost) );

    // Getting the inverse of D

    diagonal_inverse<<<1, N>>>(dev_D);
    // Finding Y
    product<<<dimGrid, dimBlock>>>(dev_D, dev_Q_t, dev_aux);
    product_v<<<1,N>>>(dev_aux, dev_b, dev_y);

    HANDLE_ERROR( cudaMemcpy(host_y, dev_y, sizeof(double) * N, cudaMemcpyDeviceToHost) );

    HANDLE_ERROR(cudaMemcpy(host_x, dev_y, N * sizeof(double), cudaMemcpyDeviceToHost));

    // Finally solving for x. First we need to include the posibility that X_N is not 1

    host_x[N - 1] = host_x[N - 1] / r[N * N - 1]; // To include the possible sing in the reduced matrix
    
    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_y, dev_x, N * sizeof(double), cudaMemcpyDeviceToDevice));

    for(int i = 1; i < N; i++) {
        double *dev_hold;
        HANDLE_ERROR(cudaMalloc((void**)&dev_hold, (i) * sizeof(double)));
        solver<<<1, i>>>(dev_R, dev_x, dev_y, dev_hold, i);
    }

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "\nTime taken:  %3.10f ms\n", elapsedTime );

    HANDLE_ERROR(cudaMemcpy(host_x, dev_x, N * sizeof(double), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_input));
	HANDLE_ERROR(cudaFree(dev_m));
    HANDLE_ERROR(cudaFree(dev_R));
    HANDLE_ERROR(cudaFree(dev_A));

    printf("\nOriginal Extended Matrix:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", a[i * N + j]);
        }
        printf("\t %f\n", b[i]);
    }

    printf("\nQ Matrix:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", input[i * N + j]);
        }
        printf("\n");
    }

    printf("\nR Matrix:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%f ", r[i * N + j]);
        }
        printf("\n");
    }

    //printf("\nD Matrix:\n");
    //for(int i = 0; i < N; i++) {
        //for(int j = 0; j < M; j++) {
            //printf("%f ", host_d[i * N + j]);
        //}
        //printf("\n");
    //}

    //printf("\nY Vector:\n");
    //for(int i = 0; i < N; i++) {
        //printf("%f ", host_y[i]);
    //}
    //printf("\n");

    printf("\nX Solutions:\n");
    for(int i = 0; i < N; i++) {
        printf("%f ", host_x[i]);
    }
    printf("\n");
}