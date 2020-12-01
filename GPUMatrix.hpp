#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <random>

#include "Templates.hpp"
#include "include/IO.hpp"

class GPUMatrix : public AbstractMatrix<double> {
public:
	static cublasHandle_t* handle;
	static double SMThreads;// double as floating point operastions are later performed
	static cudaStream_t stream; 


    static void GPUSupported(bool* supported);

	GPUMatrix() { //Null constructor
		arr = nullptr;
		struct_type = StructType::STRUCT_GPUMatrix;
	}
	GPUMatrix(size_t Y, size_t X) {//standard constructor taking the number of columns in the matrix and the number of rows
			//then allocating the appropriate amount of memory on the device
		struct_type = StructType::STRUCT_GPUMatrix;
		size = X * Y;
		y = Y;
		x = X;
		allocate_GPU_memory(arr, size);
	}
	//Copy constructor
	// GPUMatrix(GPUMatrix* m) {
	// 	struct_type = StructType::STRUCT_GPUMatrix;
	// 	m->copy(this);
	// }

	

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


	
	double index(size_t Y, size_t X) const override {//Gets a value from the matrix indexed in both dimensions
		double ret = 0;
		cpy(&ret, &arr[getIndex(Y,X)], sizeof(double), cudaMemcpyDeviceToHost); //copy's the value from GPU memory to host memory
		return ret;
	}
	double index(size_t X) const override {//Gets a value from the matrix indexed By a single dimension 0 to Matrix size
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

	double sum() const override;

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
	GPUMatrix* copy() const override {
		GPUMatrix* C = new GPUMatrix(y, x);
		copy(C);
		return C;
	}
	/**
	 * @param a a copy of the matrix on the same device
	 */
	void copy(AbstractMatrix* m) const  override {
		if(dynamic_cast<const GPUMatrix*>(m) == NULL){
			ilog(ERROR, "Cannot copy NULL or GPUMatrix to non-GPUMatrix");
		}
		cpy(m->arr, arr, size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
		m->x = x;
		m->y = y;
	}
	
	void copyToCPU(double* CPUPtr) const {
		cpy(CPUPtr, arr, size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}
	double* copyToCPU() const {
		double* CPUPtr = new double[size];// (double*)malloc(size*sizeof(double));
		cpy(CPUPtr, arr, size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		return CPUPtr;
	}

	void print() const override;
	void print(int resolution) const override { //print the matrix to the console with a defined number of digits
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
	double* copy_to_CPU() const override{
		return copyToCPU();
	}

	/**
	 * @return a copy of the matrix stored in CPU memory
	 */
	const double* get_CPU_pointer() const override{
		return copyToCPU();
	}


	void convolute(AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* out, int outY, int outX, int outZ, int convY, int convX, int convZ) override;

	void convBackprop(AbstractMatrix* in, AbstractMatrix* layer, AbstractMatrix* this_layer_conv_error, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* out, AbstractMatrix* out_error, AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) override;



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