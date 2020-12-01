#include "Matrix.hpp"



bool Matrix::checked;
bool Matrix::hasGPU;
std::random_device rd;
std::mt19937 Matrix::mt = std::mt19937(rd());



bool Matrix::checkGPU() {
	if (checked) return hasGPU;
	else {
		GPUMatrix::GPUSupported(&hasGPU);
		checked = true;
		return hasGPU;
	}
	
}


void Matrix::forceUseGPU() {
	Matrix::checkGPU();
	hasGPU = true;
}
void Matrix::forceUseCPU() {
	Matrix::checkGPU();
	hasGPU = false;
}




