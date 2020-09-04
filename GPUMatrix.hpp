#pragma once
//#pragma comment(lib,"cublas.lib") 
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "IO.hpp"
#include <random>
#include <iostream>
#include <complex>
#include <chrono>
#include <math.h>
#include <vector>
#include "Templates.hpp"

class GPUMatrix : public AbstractMatrix<double> {
public:
	static cublasHandle_t* handle;
	static double SMThreads;// double as floating point operastions are later performed
	static cudaStream_t stream; 
	void alloc(double* ptr, size_t size) {
		auto err = cudaMalloc(&arr, size * sizeof(double));
		if (err != cudaSuccess) {
			while (err != cudaSuccess) {
				std::cout << cudaGetErrorString(err) << std::endl;
				cudaMalloc(&arr, size * sizeof(double));
			}
		}
	}
	void cpy(double* dst, double* src, size_t size, cudaMemcpyKind kind) {
		auto err = cudaMemcpy(dst, src, size, kind);
		while (err != cudaSuccess) {
			if((dst == 0) || (src == 0)) ilog(FATAL_ERROR, "dstination or source pointer NULL");
			std::cout << "error while copying on GPU: " << err;
			err = cudaMemcpy(dst, src, size, kind);
		}
	}
	GPUMatrix() { //Null constructor
		arr = nullptr;
	}
	GPUMatrix(size_t Y, size_t X) {//standard constructor taking the number of columns in the matrix and the number of rows
			//then allocating the appropriate amount of memory on the device
		size = X * Y;
		y = Y;
		x = X;
		alloc(arr, size);
	}
	//Copy constructor
	GPUMatrix(GPUMatrix* m) {
		m->copy(this);
	}

	enum MEM { CPU, GPU };

	/**
	 * Construct CPU matrix, using existing array
	 * Data will NOT be copied if array_location = GPU
	 * @param Y Y dimension
	 * @param X X dimension
	 * @param arr existing array to use, assumed of size Y*X
	 * @param array_location the Memory type the pointer is pointing to, CPU/GPU/...
	 */
	GPUMatrix(size_t Y, size_t X, double* arr, MEM array_location) {
		y = Y;
		x = X;
		size = x * y;
		if (this->arr != 0) {
			cudaFree(this->arr);//TODO remove if this is never triggered
		}
		if(array_location == GPU){
			this->arr = arr;
		}
		else{
			alloc(this->arr, size);
			cpy(this->arr, arr, size*sizeof(double), cudaMemcpyHostToDevice);
		}
	}


	
	double index(size_t Y, size_t X) override {//Gets a value from the matrix indexed in both dimensions
		double ret = 0;
		cpy(&ret, &arr[getIndex(Y,X)], sizeof(double), cudaMemcpyDeviceToHost); //copy's the value from GPU memory to host memory
		return ret;
	}
	double index(size_t X) override {//Gets a value from the matrix indexed By a single dimension 0 to Matrix size
		double ret = 0;
		cpy(&ret, &arr[X], sizeof(double), cudaMemcpyDeviceToHost); //copy's the value from GPU memory to host memory
		return ret;
	}


	void setIndex(size_t Y, size_t X, double value) override { //set the value of an co-ordinate in the GPU matrix By copying the parameter from the CPU to the GPU device at that memory location
		double temp = value;
		cudaMemcpy(&arr[getIndex(Y,X)], &temp, sizeof(double), cudaMemcpyHostToDevice);
	}
	void setIndex(size_t i, double value) override { //set the value of an index in the GPU matrix arr By copying the parameter from the CPU to the GPU device at that memory location
		double temp = value;
		cudaMemcpy(&(arr[i]), &temp, sizeof(double), cudaMemcpyHostToDevice);
	}

	void addIndex(size_t Y, size_t X, double value) override {
		double temp = value + index(Y, X);
		cudaMemcpy(&arr[getIndex(Y,X)], &temp, sizeof(double), cudaMemcpyHostToDevice);
	}
	void addIndex(size_t i, double value) override { //set the value of an index in the GPU matrix arr By copying the parameter from the CPU to the GPU device at that memory location
		double temp = value + index(i);
		cudaMemcpy(&(arr[i]), &temp, sizeof(double), cudaMemcpyHostToDevice);
	}

	GPUMatrix* multiply(AbstractMatrix* B) override;
	void multiply(AbstractMatrix* B, AbstractMatrix* C) override;
	GPUMatrix* multiplyA(AbstractMatrix* B) override;
	void multiplyA(AbstractMatrix* B, AbstractMatrix* C) override;
	GPUMatrix* multiplyB(AbstractMatrix* B) override;
	void multiplyB(AbstractMatrix* B, AbstractMatrix* C) override;
	GPUMatrix* multiplyAB(AbstractMatrix* B) override;
	void multiplyAB(AbstractMatrix* B, AbstractMatrix* C) override;

	GPUMatrix* multiplyElementWise(AbstractMatrix* B) override;
	void multiplyElementWise(AbstractMatrix* B, AbstractMatrix* C) override;

	GPUMatrix* divideElementWise(AbstractMatrix* B) override;
	void divideElementWise(AbstractMatrix* B, AbstractMatrix* C) override;

	GPUMatrix* sigmoid() override;
	void sigmoid(AbstractMatrix* C) override;

	GPUMatrix* sigmoidDifferential() override;
	void sigmoidDifferential(AbstractMatrix* C) override;

	GPUMatrix* add(AbstractMatrix* B) override;
	void add(AbstractMatrix* B, AbstractMatrix* C) override;

	void addAssign(AbstractMatrix* B) override;
	
	GPUMatrix* subtract(AbstractMatrix* B) override;
	void subtract(AbstractMatrix* B, AbstractMatrix* C) override;

	void subtractAssign(AbstractMatrix* B) override;

	GPUMatrix* addConst(double B) override;
	void addConst(double B, AbstractMatrix* C) override;

	GPUMatrix* scale(double B) override;
	void scale(double B, AbstractMatrix* C) override;

	void randomFill(double min, double mAx) override;

	void randomFill(double negmin, double negmAx, double min, double mAx) override;

	void fill(double value) override { //fills the matrix with a given value, needs conversion to kernel 
		cudaMemset2D(arr, sizeof(double), value, x, y);
	}
	
	void transpose() override { //transposes a GPU matrix using the CPU, needs GPU conversion
		GPUMatrix* C = transposeNew();
		this->x = C->y;
		this->y = C->x;
		cudaFree(this->arr);
		this->arr = C->arr;
		C->arr = NULL;
	}
	GPUMatrix* transposeNew() override { //transposes a GPU matrix using the CPU, needs GPU conversion
		GPUMatrix* C = new GPUMatrix(x, y);
		transpose(C);
		return C;
	}
	/**
	 * @return a copy of the matrix on the same device
	 */
	GPUMatrix* copy() override {
		GPUMatrix* C = new GPUMatrix(x, y);
		copy(C);
		return C;
	}
	/**
	 * @param a a copy of the matrix on the same device
	 */
	void copy(AbstractMatrix* m) override {
		cpy(m->arr, arr, size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
		x = m->x;
		y = m->y;
	}
	
	void copyToCPU(double* CPUPtr) {
		cpy(CPUPtr, arr, size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}
	double* copyToCPU() {
		double* CPUPtr = (double*)malloc(size*sizeof(double));
		cpy(CPUPtr, arr, size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		return CPUPtr;
	}

	void print() override;
	void print(int resolution) override { //print the matrix to the console with a defined number of digits
		std::streamsize ss = std::cout.precision();
		std::cout.precision(resolution);
		for (int c = 0; c < y; c++) {
			std::cout << std::fixed << index(c, 0);
			for (int r = 1; r < x; r++) {
				std::cout << std::fixed << ",		" << index(c, r);
			}
			std::cout << std::endl;
		}
		std::cout.precision(ss);
	}


	/**
	 * @return a copy of the matrix stored in CPU memory
	 */
	double* get_CPU_pointer(){
		return copyToCPU();
	}
	/**
	 * @return the matrix stored in CPU memory - warning this may be a direct refernce to the Matrices array
	 */
	double* get_CPU_pointer_read_only(){
		return copyToCPU();
	}

	void convolute(AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* net,  AbstractMatrix* out, int inX, int inY, int inZ, int outX, int outY, int outZ, int convX, int convY) override;

	void convBackprop(AbstractMatrix* in, AbstractMatrix* layer, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* net, AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) override;



	void deconstruct() override {
		if (arr) {
			cudaFree(arr);
			arr = nullptr;
		}
	}

	/**
	 * used to de-refence array to prevent freeing during deconstruction
	 */
	void delete_array_ref() override {
		arr = nullptr;
	}





	
	private:
		void transpose(GPUMatrix* B);
};