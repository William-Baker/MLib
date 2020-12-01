#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "include/IO.hpp"
#include "GPUTensor.hpp"

/* void GPUTensor::convolute(AbstractTensor* layer, AbstractTensor* bias, AbstractTensor* out){
    double proportion = sqrt(GPUMatrix::SMThreads / (out->x * out->y * out->z));
    dim3 Thread(std::max(out->x * proportion, 1), std::max(out->y * proportion), std::max(out->z * proportion));
    dim3 Block(ceil((double)out->x / Thread.x), ceil((double)out->y / Thread.y), ceil((double)out->z / Thread.z));
    convolute_kernel << < Block, Thread , 0, stream>> > (this->get_implementation()->get_static_array(),
     layer->get_implementation()->get_static_array(),
     bias->this->get_implementation()->get_static_array(),
     out->get_implementation()->get_static_array(), ); //all the sizes);

} */

namespace GPUTensorKernels{
    __device__ __forceinline__ long indexing(int X, int Y, int y_dim){
		return AbstractTensor::indexing(X, Y, y_dim);
	}

    __device__ __forceinline__ long indexing(int X,int Y,int Z,int y_dim,int z_dim){
		return AbstractTensor::indexing(X, Y, Z, y_dim, z_dim);
	}

    __device__ __forceinline__ long indexing(int X, int Y, int Z, int A, int y_dim, int z_dim, int a_dim){
        return AbstractTensor::indexing(X, Y, Z, A, y_dim, z_dim, a_dim);
	}
	
    __device__ __forceinline__ long indexing(int X, int Y, int Z, int A, int B, int y_dim, int z_dim, int a_dim, int b_dim){
		return AbstractTensor::indexing(X, Y, Z, A, B, y_dim, z_dim, a_dim, b_dim);
	}
}