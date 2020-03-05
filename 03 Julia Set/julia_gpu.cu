/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "book.h"
#include "cpu_bitmap.h"

#define DIM 1000

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ float julia( int x, int y ) {
    const float scale = -2.0;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    //cuComplex c(-0.8, 0.156); // IM01
    //cuComplex c(-0.4, 0.6); // IM02
    //cuComplex c(-1.0, 0.0); // IM03
    //cuComplex c(0.25, 0.0); // IM04
    //cuComplex c(jx, jy); // Mandelbrot
    //cuComplex c(-1.77578, 0.0); // IM06
    //cuComplex c(0.285, 0.0); // IM07
    //cuComplex c(0.285, 0.01); // IM08
    cuComplex c(-0.6, 0.0); // IM09

    cuComplex z(jx, jy);
    //cuComplex z(0.0, 0.0);

    int i = 0;
    for (i=0; i < 2000; i++) {
        z = z * z + c;
        if (z.magnitude2() > 100)
            return i;
    }

    return z.magnitude2();
}

__global__ void kernel( unsigned char *ptr ) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    float juliaValue = julia( x, y );
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0 * juliaValue;
    ptr[offset*4 + 2] = 128 * juliaValue;
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;

    dim3    grid(DIM,DIM);
    kernel<<<grid, 1>>>( dev_bitmap );

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
                              
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
                              
    bitmap.display_and_exit();
}

