//#pragma once
#include "Matrix.hpp"

#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

cublasHandle_t* GPUMatrix::handle;
double GPUMatrix::SMThreads;
bool Matrix::checked;
bool Matrix::pHasGPU;
std::random_device rd;
std::mt19937 Matrix::mt = std::mt19937(rd());

cudaStream_t GPUMatrix::stream; 

bool Matrix::checkGPU() {
	if (checked) return pHasGPU;
	else {
		pHasGPU = GPUStats();
		checked = true;
		return pHasGPU;
	}
}
bool Matrix::GPUStats() {
	GPUMatrix::handle = new cublasHandle_t();
	cublasStatus_t status = cublasCreate(GPUMatrix::handle);

	if (*GPUMatrix::handle == NULL && status != CUBLAS_STATUS_SUCCESS) {
		pHasGPU = false;
	}
	else {
		pHasGPU = true;
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		for (int i = 0; i < nDevices; i++) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			GPUMatrix::SMThreads = prop.maxThreadsPerBlock;
		}
		//cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
		cudaStreamCreateWithPriority(&GPUMatrix::stream, cudaStreamDefault, 0);
		//cudaStreamCreateWithPriority(&GPUMatrix::stream, cudaStreamNonBlocking, 0);
		//cudaStreamCaptureMode cap = cudaStreamCaptureModeGlobal;
		cudaStreamCaptureMode cap = cudaStreamCaptureModeThreadLocal;

		cudaThreadExchangeStreamCaptureMode(&cap);
	}

	return pHasGPU;
}

void Matrix::forceUseGPU() {
	Matrix::hasGPU();
	pHasGPU = true;
}
void Matrix::forceUseCPU() {
	Matrix::hasGPU();
	pHasGPU = false;
}
























































//=========================================================== CPU Matrix ==========================================================================

void CPUMatrix::randomFill(double min, double max) {
	std::uniform_real_distribution<double> RandDist(min, max);
	for (int index = 0; index < size; index++) {
		arr[index] = RandDist(Matrix::mt);
	}
};


void CPUMatrix::randomFill(double lowerMin, double lowerMax, double upperMin, double upperMax) {
	std::uniform_real_distribution<double> SignRandDist(0, 2);
	std::uniform_real_distribution<double> RandDistLower(lowerMin, lowerMax);
	std::uniform_real_distribution<double> RandDistUpper(upperMin, upperMax);
	for (int index = 0; index < size; index++) {
		if (SignRandDist(Matrix::mt) > 1) {
			arr[index] = RandDistLower(Matrix::mt);
		}
		else {
			arr[index] = RandDistUpper(Matrix::mt);
		}
	}
}

void CPUMatrix::convBackprop(AbstractMatrix* in, AbstractMatrix* layer, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* net, AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) {
	/*
	for (int off = 0; off < outZ*outZ; off+= outZ) {
		for (int oX = 0; oX < outX; oX++) {
			for (int oY = 0; oY < outY ; oY++) {
				for (int cX = 0; cX < convX; cX++) {
					for (int cYZ = 0; cYZ < convY * convZ; cYZ++) {
						prevError->addIndex(oY + cYZ, oX + cX, layer->index(cYZ, cX + off ) * index(oY, oX));
						//layer->addIndex(cYZ, cX, in->index(oY + cYZ, oX + cX) * index(oY, oX ));
					}
				}
			}
		}
	}
	*/
	double biasScale = 1 / (convX * convY);
	//Matrix gradient(net->y, net->x);
	for (int x = 0; x < net->size; x++) {
		gradient->setIndex(x, index(x) * tanhd(net->index(x)));

	}
	prevError->fill(0);
	for (int f = 0; f < outZ; f++) {
		for (int oX = 0; oX < outX; oX++) {
			for (int oY = 0; oY < outY; oY++) {
				for (int cX = 0; cX < convX; cX++) {
					for (int cYZ = 0; cYZ < convY * convZ; cYZ++) {
						//prevError->addIndex(oY + cYZ, oX + cX, layer->index(cYZ, cX * outZ + f) * gradient.index(oY * outZ + f, oX));
						prevError->addIndex(oY + cYZ, oX + cX, layer->index(cYZ, cX * outZ + f) * index(oY * outZ + f, oX));
						double temp = gradient->index(oY, oX * outZ + f) * in->index(oY + cYZ, oX + cX) * LR;
						layer->addIndex(cYZ, cX * outZ + f, temp);
						bias->addIndex(f, temp * biasScale);
					}
				}
			}
		}
	}


	/*layer->fill(0);
	for (int oX = 0; oX < outX; oX++) {
		for (int oY = 0; oY < outY; oY++) {
			for (int cX = 0; cX < convX; cX++) {
				for (int cYZ = 0; cYZ < convY * convZ; cYZ++) {
					for (int f = 0; f < outZ; f++) {
						layer->addIndex(cYZ, cX * outZ + f, index(oY, oX * outZ + f) * in->index(oY + cYZ, oX + cX));
					}
				}
			}
		}
	}*/
	//gradient.~Matrix();

}


//=================================================================================================================================================







































std::chrono::time_point<std::chrono::high_resolution_clock> Timer::startTime;
void Timer::start() {
	startTime = std::chrono::high_resolution_clock::now();
}
size_t Timer::time() {
	auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
	return std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
}