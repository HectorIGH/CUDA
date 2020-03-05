#include <stdio.h>

__global__ void hello() {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	printf("Hello wolrd! from thread %d\n", tid);
}

int main(){
	int NUMBER_OF_BLOCKS = 2;
	int NUMBER_OF_THREADS = 10;
	hello<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS>>>();
	return 0;
}