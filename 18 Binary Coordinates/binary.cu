/*
* Compile using following structure
* nvcc -rdc=true -arch compute_35 binary.cu -o bn
* and profile with nvprof --unified-memory-profiling off ./bn
* check memory error with cuda-memcheck ./bn
*/
#include <cuda.h>
#include "book.h"
#include <cuda_runtime.h>


__global__ void binarize(double *N, double *binary) {
    int tid = threadIdx.x;

    binary[tid] = 1 << tid & int(N[0]) ? 1 : 0;

}

__global__ void residuals_coordinates(double *N, double *x, double *residuals) {
    int tid = threadIdx.x;

    residuals[tid] = int(powf(x[0], 1 << tid)) % int(N[0]) ;

    //printf("In residuals %f\n\n", powf(x[0], 1 << tid));

}

__global__ void masking(double *binary, double *residuals) {
    int tid = threadIdx.x;

    if(binary[tid] == 0.0) {
        __syncthreads();
        residuals[tid] = 1;
    }
}

__global__ void residual(double *residuals, double *m) {
    int tid = threadIdx.x;
    int number_of_threads = blockDim.x;
    int step_size = 1;

    while(number_of_threads > 0) {
        if(tid < number_of_threads) {
            
            int first = tid * 2 * step_size;
            int second = first + step_size;

            residuals[first] *= residuals[second];
            __syncthreads();
        }
        step_size <<= 1;
        number_of_threads >>= 1;
    }
    residuals[0] = int(residuals[0]) % int(m[0]);
}


int main( int argc, char *argv[] ) {
    
    double host_x[1] = {6};
    double host_e[1] = {15};
    double host_m[1] = {21};

    int bits = int(log2(host_e[0])) + 1;

    int power_bits = 1;
    while (power_bits < bits){
        power_bits <<= 1;
    }
    bits = power_bits;
    
    double *host_binary;
    double *host_residuals;
    double *host_products;
    double *host_residuo;

    host_binary = (double*)malloc(sizeof(double) * bits);
    host_residuals = (double*)malloc(sizeof(double) * bits);
    host_products = (double*)malloc(sizeof(double) * bits);
    host_residuo = (double*)malloc(sizeof(double) * 1);

    //srand((unsigned) time(NULL));


    double *dev_x;
    double *dev_e;
    double *dev_m;
    
    double *dev_binary;
	double *dev_residuals;
    double *dev_products;

    HANDLE_ERROR(cudaMalloc((void**)&dev_x, 1 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_e, 1 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_m, 1 * sizeof(double)));
    
    HANDLE_ERROR(cudaMalloc((void**)&dev_binary, bits * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_residuals, bits * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_products, bits * sizeof(double)));

    HANDLE_ERROR(cudaMemcpy(dev_x, host_x, 1 * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_e, host_e, 1 * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_m, host_m, 1 * sizeof(double), cudaMemcpyHostToDevice));



    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    
    binarize<<<1, bits>>>(dev_e, dev_binary);

    residuals_coordinates<<<1, bits>>>(dev_m, dev_x, dev_residuals);

    HANDLE_ERROR(cudaMemcpy(dev_products, dev_residuals, bits * sizeof(double), cudaMemcpyDeviceToDevice));

    masking<<<1, bits>>>(dev_binary, dev_products);

    residual<<<1, bits / 2>>>(dev_products, dev_m);

	
    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "\nTime taken:  %3.10f ms\n", elapsedTime );


    HANDLE_ERROR(cudaMemcpy(host_binary, dev_binary, bits * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_residuals, dev_residuals, bits * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_products, dev_products, bits * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_residuo, dev_products, 1 * sizeof(double), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_x));
    HANDLE_ERROR(cudaFree(dev_e));
    HANDLE_ERROR(cudaFree(dev_m));
    HANDLE_ERROR(cudaFree(dev_binary));
    HANDLE_ERROR(cudaFree(dev_residuals));
    HANDLE_ERROR(cudaFree(dev_products));

    printf("\nBinary Coordinates Representation of %1.0f:\n\n", host_e[0]);
    for(int i = 0; i < bits; i++) {
        printf("%1.0f ", host_binary[i]);
    }

    printf("\n\nResiduals Coordinates:\n\n");
    for(int i = 0; i < bits; i++) {
        printf("%f ", host_residuals[i]);
    }

    printf("\nProducts:\n");
    for(int i = 0; i < bits; i++) {
        printf("%f ", host_products[i]);
    }

    printf("\n\n[%1.0f ^ %1.0f]_%1.0f = %f\n", host_x[0],host_e[0], host_m[0], host_residuo[0]);
}