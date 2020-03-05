/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 matrix_qr.cu -o qr
* and profile with nvprof --unified-memory-profiling off ./qr
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>

extern const int N = 8;		//this defines the number of elements in each vector
extern const int M = 8;		//this defines the number of vectors that need to be orthogonalized

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

__global__ void innerprod_self(double *in, double *a, int st){
	__shared__ double prod[N];
	int k = threadIdx.x;
	prod[k] = in[st + k]*in[k + st];
	__syncthreads();
	
	if(k==0){
		double temp=0;
		for(int e=0;e<N;e++){
			temp = temp + prod[e];	
		}	
		a[0] = sqrt(temp);
	}
}

__global__ void scale(double *in, double *val, int st){
	int j = threadIdx.x;
	in[j+st] = in[j+st] / val[0];
}

__global__ void calcCoef(double *v,double *coef, int prev_vindex, int i){
	__shared__ double prod[N];
	int r = blockIdx.x;
	int t = threadIdx.x;
	prod[t] = v[prev_vindex + t]*v[N*(r+i) + t];
	__syncthreads();
	if(t==0){
		double temp;
		temp  =0;
		for(int e=0;e<N;e++){
			temp = temp + prod[e];	
		}
		coef[blockIdx.x] = -1*temp;
	}
}

__global__ void combine(double *in, double *coe, int prev, int cur){
	int th = threadIdx.x;
	int bl = blockIdx.x;
	int cu = (bl+cur)*N;
	in[cu + th] = in[cu + th] + coe[bl]*in[prev+th];
}


int main( void ) {
    // Input in column order
    //double input[N * N] = {12,6,-4,-51,167,24,4,-68,-41};
    //double a[N * N] = {12,6,-4,-51,167,24,4,-68,-41};
    //double r[N * N] = {12,6,-4,-51,167,24,4,-68,-41};

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
            input[i * N + j] = rand() % 10;
            a[i * N + j] = input[i * N + j];
            r[i * N + j] = input[i * N + j];
        }
        b[i] = rand() % 10;
    }



    double *host_d;
    double *host_y;
    double *host_x;

    host_d = (double*)malloc(sizeof(double) * M * N);
    host_y = (double*)malloc(sizeof(double) * N);
    host_x = (double*)malloc(sizeof(double) * N);

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
	int prev_startindex;

	HANDLE_ERROR(cudaMalloc((void**)&dev_input, (M*N)*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_m, sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_R, (M*N)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_Q_t, (M*N)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_D, (M*N)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_A, (M*N)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_aux, (M*N)*sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_y, N * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_x, N * sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(dev_input,input,(M*N)*sizeof(double), cudaMemcpyHostToDevice));

    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );


	for(int w = 1; w < M; w++){
		double *dev_coef;
		int num_of_coe;
		num_of_coe = M - w;
		HANDLE_ERROR(cudaMalloc((void**)&dev_coef, (M-w)*sizeof(double)));
		prev_startindex = (w-1) * N;

		innerprod_self<<<1,N>>>(dev_input,dev_m, prev_startindex);
		scale<<<1,N>>>(dev_input,dev_m,prev_startindex);

		calcCoef<<<num_of_coe,N>>>(dev_input, dev_coef, prev_startindex,w);

		combine<<<num_of_coe,N>>>(dev_input, dev_coef, prev_startindex, w);

		HANDLE_ERROR(cudaFree(dev_coef));
	}

	prev_startindex = (M-1)*N;
	innerprod_self<<<1,N>>>(dev_input,dev_m, prev_startindex);
	scale<<<1,N>>>(dev_input,dev_m,prev_startindex);
	
	HANDLE_ERROR(cudaMemcpy(input, dev_input, M*N*sizeof(double), cudaMemcpyDeviceToHost));
    // Here input and dev_input holds the transpose of Q. Q^T

    HANDLE_ERROR(cudaMemcpy(dev_A, a,(M*N)*sizeof(double), cudaMemcpyHostToDevice));
    traspose<<<1, 1>>>(dev_A);
    dim3 dimBlock(N, N);
    dim3 dimGrid(1, 1);
    product<<<dimGrid, dimBlock>>>(dev_input, dev_A, dev_R);

    HANDLE_ERROR( cudaMemcpy(r, dev_R, sizeof(double) * M * N, cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(dev_Q_t, dev_input, sizeof(double) * M * N, cudaMemcpyDeviceToDevice));
    
    // Calculating Q_t * Q = D
    //traspose<<<1, 1>>>(dev_Q_t); // Not necessary since dev_input and hence dev_Q_t already holds the transpose.
    traspose<<<1, 1>>>(dev_input); // We get the real Q matrix
    product<<<dimGrid, dimBlock>>>(dev_Q_t, dev_input, dev_D);

    HANDLE_ERROR( cudaMemcpy(host_d, dev_D, sizeof(double) * M * N, cudaMemcpyDeviceToHost) );

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
        for(int j = 0; j < M; j++) {
            printf("%f ", a[i + j * N]);
        }
        printf("\t %f\n", b[i]);
    }

    printf("\nQ Matrix:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", input[i * N + j]);
        }
        printf("\n");
    }

    printf("\nR Matrix:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", r[i * N + j]);
        }
        printf("\n");
    }

    //printf("\nD Matrix:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            //printf("%f ", host_d[i * N + j]);
        }
        //printf("\n");
    }

    //printf("\nY Vector:\n");
    for(int i = 0; i < N; i++) {
        //printf("%f ", host_y[i]);
    }
    //printf("\n");

    printf("\nX Solutions:\n");
    for(int i = 0; i < N; i++) {
        printf("%f ", host_x[i]);
    }
    printf("\n");
}