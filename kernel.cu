#pragma once


#include <cuda_runtime.h>
#include "device_launch_parameters.h"
//#include <math.h>
#include <cuda_runtime.h>

//TODO remember all the XD's switched to YD's
__device__ __forceinline__ long Index(int y, int x, int YD)
{
	return x * YD + y;
}


//__device__ __forceinline__ double tanh(double x) {
//	double ex = exp(x);
//	double nex = exp(-x);
//	return (exp(x) - exp(-x)) / (exp(-x) + exp(x));
//}

//CY = AY
//CX = BX
//AX = BY INNER
__global__ void GPUMultKernel(double* A, double* B, double* C, int CY, int CX, int AX) {
	int Cy = blockDim.x * blockIdx.x + threadIdx.x; //col 2
	int Cx = blockDim.y * blockIdx.y + threadIdx.y; //row 1
	if ((Cy < CY) & (Cx < CX)) {
		C[Index(Cy, Cx, CY)] = 0;
		for (int Ax = 0; Ax < AX; Ax++) {
			C[Index(Cy, Cx, CY)] += A[Index(Cy, Ax, CY)] * B[Index(Ax, Cx, AX)];
		}
	}
}

//CY = AX
//CX = BX
//AY = BY INNER
__global__ void GPUMultKernelA(double* A, double* B, double* C, int CY, int CX, int AY) {
	int Cy = blockDim.x * blockIdx.x + threadIdx.x; //col 2
	int Cx = blockDim.y * blockIdx.y + threadIdx.y; //row 1
	if ((Cy < CY) & (Cx < CX)) {
		C[Index(Cy, Cx, CY)] = 0;
		for (int Ay = 0; Ay < AY; Ay++) {
			C[Index(Cy, Cx, CY)] += A[Index(Ay, Cy, AY)] * B[Index(Ay, Cx, AY)];
		}
	}
}

//CY = AY
//CX = BY
//AX = BX INNER
__global__ void GPUMultKernelB(double* A, double* B, double* C, int CY, int CX, int AX) {
	int Cy = blockDim.x * blockIdx.x + threadIdx.x; //col 2
	int Cx = blockDim.y * blockIdx.y + threadIdx.y; //row 1
	if ((Cy < CY) & (Cx < CX)) {
		C[Index(Cy, Cx, CY)] = 0;
		for (int Ax = 0; Ax < AX; Ax++) {
			C[Index(Cy, Cx, CY)] += A[Index(Cy, Ax, CY)] * B[Index(Cx, Ax, CX)];
		}
	}
}

//CY = AX
//CX = BY
//AY = BX INNER
__global__ void GPUMultKernelAB(double* A, double* B, double* C, int CY, int CX, int AY) {
	int Cy = blockDim.x * blockIdx.x + threadIdx.x; //col 2
	int Cx = blockDim.y * blockIdx.y + threadIdx.y; //row 1
	if ((Cy < CY) & (Cx < CX)) {
		C[Index(Cy, Cx, CY)] = 0;
		for (int Ay = 0; Ay < AY; Ay++) {
			C[Index(Cy, Cx, CY)] += A[Index(Ay,Cy, AY)] * B[Index(Cx, Ay, CX)];
		}
	}
}

__global__ void GPUTranspose(double* A, double* B, int AY, int AX) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y= blockDim.y * blockIdx.y + threadIdx.y;
	if ((x < AX) & (y< AY)) {
		B[Index(x, y, AX)] = A[Index(y, x, AY)];
	}
}

__device__ __forceinline__ double GPUSigmoidKernel(double a)
{
	return 1.0 / (1.0 + exp(-a));
}

__global__ void GPUSigmoid(double* A, double* C, long size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		C[i] = GPUSigmoidKernel(A[i]);
	}
}

__device__ __forceinline__ double GPUSigmoidDifferentialKernel(double a)
{
	return a * (1.0 - a);
}

__global__ void GPUSigmoidDifferential(double* A, double* C, long size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		C[i] = GPUSigmoidDifferentialKernel(A[i]);
	}
}

__global__ void GPUAddConst(double* A, double B, double* C, long size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		C[i] = A[i] + B;
	}
}

__global__ void GPUElementwiseAdd(double* A, double* B, double* C, long size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		C[i] = A[i] + B[i];
	}
}

__global__ void GPUAddAssign(double* A, double* B, long size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		A[i] += B[i];
	}
}

__global__ void GPUElementwiseSubtract(double* A, double* B, double* C, long size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		C[i] = A[i] - B[i];
	}
}

__global__ void GPUSubtractAssign(double* A, double* B, long size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		A[i] -= B[i];
	}
}

__global__ void GPUElementWiseMultiply(double* A, double* B, double* C, long size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		C[i] = A[i] * B[i];
	}
}

__global__ void GPUElementWiseDivide(double* A, double* B, double* C, long size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		C[i] = A[i] / B[i];
	}
}

__global__ void GPUScale(double* A, double B, double* C, long size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		C[i] = A[i] * B;
	}
}

__global__ void ConvKernel(double* in, double* layer, double* bias, double* net, double* out, int inY, int inZ,int outX, int outY, int outZ, int convY, int convX) {
	int fX = blockDim.x * blockIdx.x + threadIdx.x; //
	int fY = blockDim.y * blockIdx.y + threadIdx.y; //
	int filter = blockDim.z * blockIdx.z + threadIdx.z; //

	/*if ((fX < outX) & (fY < outY) & (filter < outZ)) {
		double temp = 0;
		for (int cX = 0; cX < convX; cX++) {
			for (int cY = 0; cY < convY * inZ; cY++) {
				temp += in[Index(fY * inZ + cY, fX + cX, inY*inZ)] * layer[Index(cY, cX + filter * convX, convY * inZ)];
			}
		}
		temp += bias[filter];
		double ex = exp(temp);
		double nex = exp(temp);
		temp = (ex - nex) / (nex + ex);
		out[Index(fY * outZ + filter, fX, outY*outZ)] = temp;
	}*/
	if ((fX < outX) & (fY < outY) & (filter < outZ)) {
		double temp = 0;
		for (int cX = 0; cX < convX; cX++) {
			for (int cY = 0; cY < convY * inZ; cY++) {
				temp += in[Index(fY * inZ + cY, fX + cX, inY * inZ)] * layer[Index(cY, cX + filter * convX, convY * inZ)];
			}
		}
		temp += bias[filter];
		net[Index(fY * outZ + filter, fX, outY * outZ)] = temp;
		double ex = exp(temp);
		double nex = exp(-temp);
		temp = (ex - nex) / (nex + ex);
		
		out[Index(fY * outZ + filter, fX, outY * outZ)] = temp;
	}
}

__global__ void convBackpropErrorsKernel(double* gradient, double* net, double* outError, int netSize) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < netSize) {
		double ex = exp(net[i]);
		double nex = exp(-net[i]);
		double temp = ex + nex;
		temp =  4 / (temp * temp);
		gradient[i] = outError[i] * temp;
	}

}

__global__ void convBackpropKernel(double* outError, double* in, double* layer, double* prevError, double* bias, double* net, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR, double* gradient, double biasScale, double inYA) {
	int oX = blockDim.x * blockIdx.x + threadIdx.x; //
	int oY = blockDim.y * blockIdx.y + threadIdx.y; //
	int f = blockDim.z * blockIdx.z + threadIdx.z; //

	if ((oX < outX) & (oY < outY) & (f < outZ)) {

		for (int cX = 0; cX < convX; cX++) {
			for (int cYZ = 0; cYZ < convY * convZ; cYZ++) {
				//prevError->addIndex(oY + cYZ, oX + cX, layer->index(cYZ, cX * outZ + f) * gradient.index(oY * outZ + f, oX));
				prevError[Index(oY + cYZ, oX + cX, outY*outZ)] += layer[Index(cYZ, cX * outZ + f, convY * convZ)] * outError[Index(oY * outZ + f, oX, outY*outZ)];
				double temp = gradient[Index(oY, oX * outZ + f, outY*outZ)] * in[Index(oY + cYZ, oX + cX, inYA)] * LR;
				layer[Index(cYZ, cX * outZ + f, convY * convZ)] += temp;
				bias[f] += temp * biasScale;
			}
		}
	}
}
