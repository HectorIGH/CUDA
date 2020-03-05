#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 100

__global__ void vector_add(float* input) {
	const int tid = threadIdx.x;
	auto step_size = 1;
	int number_of_threads = blockDim.x;

	while (number_of_threads > 0)
	{
		if (tid < number_of_threads) // still alive?
		{
			const auto fst = tid * step_size * 2;
			const auto snd = fst + step_size;
			input[fst] += input[snd];
            __syncthreads();
		}

		step_size <<= 1; 
		number_of_threads >>= 1;
	}
}

int main(){
    float *a;
    float *d_a;
    float result;

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = i;//1.0f;
    }

    // Allocate device memory 
    cudaMalloc((void**)&d_a, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);


    // Executing kernel 
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);

    vector_add<<<grid_size, block_size>>>(d_a);
    
    // Transfer data back to host memory
    cudaMemcpy(&result, d_a, sizeof(float), cudaMemcpyDeviceToHost);

    printf("The sum is: %.2f\n", result);

    // Deallocate device memory
    cudaFree(d_a);

    // Deallocate host memory
    free(a); 
}
