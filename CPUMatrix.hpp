#pragma once
#include "Templates.hpp"
#include "include/IO.hpp"
#include <iostream>
#include <cmath>
#include <cstring>

class CPUMatrix : public AbstractMatrix<double> {
public:

//Constructors
	CPUMatrix() { //Null constructor
		struct_type = StructType::STRUCT_CPUMatrix;
		size = 0;
		y = 0;
		x = 0;
		arr = nullptr;
	}
	/**
	 * standard constructor taking the number of columns in the matrix and the number of rows
	 * then allocating the appropriate amount of memory on the device
	 */
	CPUMatrix(size_t Y, size_t X) : CPUMatrix(){
		size = X * Y;
		arr = allocate_CPU_memory<double>(size);//static_cast<double*>(malloc(size * sizeof(double)));
		y = Y;
		x = X;
	}
	/**
	 * Construct CPU matrix, using existing array
	 * Data will NOT be copied - if the source is correct
	 * @param Y Y dimension
	 * @param X X dimension
	 * @param arr existing array to use, assumed of size Y*X
	 */
	CPUMatrix(size_t Y, size_t X, double* arrIn, MEM memory_location_of_array = CPU) : CPUMatrix() {
		size = X * Y;
		y = Y;
		x = X;
		if(memory_location_of_array == CPU){
			arr = arrIn;
		}
		else{
			arr = allocate_CPU_memory<double>(size);
			copy_GPU_memory(arr, arrIn, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		}
	}

	CPUMatrix(CPUMatrix& src) = delete;

	CPUMatrix(CPUMatrix&& src) : CPUMatrix(){
		arr = src.arr;
		x = src.x;
		y = src.y;
		size = src.size;

		src.arr = 0;
		src.x = src.y = src.size = 0;
	}

	CPUMatrix(const AbstractMatrix<double>* src);


//Copy
	CPUMatrix copy() const{
		double* temp = allocate_CPU_memory<double>(size);
		copy_CPU_memory(temp, arr, size);
		return CPUMatrix(y, x, temp);
	}

	/**
	 * @return a copy of the matrix stored in host memory
	 */
	double* copy_array_host() const override{
		double* ptr = allocate_CPU_memory<double>(size);
		copy_CPU_memory(ptr, arr, size);
		return ptr;
	}
	/**
	 * @return the matrix stored in CPU memory - warning this may be a direct refernce to the Matrices array
	 */
	const double* get_array_host() const override{
		return arr;
	}

//Transfers
	void transfer(CPUMatrix&& src){
		x = src.x;
		y = src.y;
		size = src.size;
		arr = src.arr;

		src.arr = 0;
		src.x = src.y = src.size = 0;
	}
	void transfer(CPUMatrix& src){
		x = src.x;
		y = src.y;
		size = src.size;
		arr = src.arr;

		src.arr = 0;
		src.x = src.y = src.size = 0;
	}

/* 	
This should rather use Transfer or else the copy constructor
	void Transfer(AbstractMatrix<double>* src){
		if(dynamic_cast<CPUMatrix*>(src)){
			CPUMatrix* actual = static_cast<CPUMatrix*>(src);
			Transfer(*actual);
		}
		else if(dynamic_cast<GPUMatrix*>(src)){
			GPUMatrix* actual = static_cast<GPUMatrix*>(src);
			x = actual->x;
			y = actual->y;
			size = actual->get_size();
			copy_GPU_memory<double>(arr, actual->arr, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		}
		else{
			ilog(FATAL_ERROR, "unknown source for transfer");
		}
	} */


	double index(size_t Y, size_t X) const override {
		if((Y >= y) || (X >= x)){
			throw("err"); //TODO remove - this was for testing
		}
		return arr[getIndex(Y, X)];
	}
	double index(size_t i) const override {
		if(i > size){
			throw("err"); //TODO remove - this was for testing
		}
		return arr[i];
	}

	void setIndex(size_t Y, size_t X, double value) override {
		if((Y >= y) || (X >= x)){
			throw("err"); //TODO remove - this was for testing
		}
		arr[getIndex(Y, X)] = value;
	}
	void setIndex(size_t i, double value) override {
		if(i > size){
			throw("err"); //TODO remove - this was for testing
		}
		arr[i] = value;
	}

	void addIndex(size_t Y, size_t X, double value) override {
		arr[getIndex(Y, X)] += value;
	}
	void addIndex(size_t i, double value) override { //set the value of an index in the GPU matrix arr By copying the parameter from the CPU to the GPU device at that memory location
		arr[i] += value;
	}

	CPUMatrix* multiply(AbstractMatrix* B) override {
		CPUMatrix* C = new CPUMatrix(y, B->x);
		multiply(B, C);
		return C;
	}
	void multiply(AbstractMatrix* B, AbstractMatrix* C) override {
		for (size_t Ay = 0; Ay < y; Ay++) {
			for (size_t Bx = 0; Bx < B->x; Bx++) {
				C->setIndex(Ay, Bx, index(Ay, 0) * B->index(0, Bx));
				for (size_t Ax = 1; Ax < x; Ax++) {
					C->addIndex(Ay, Bx, index(Ay, Ax) * B->index(Ax, Bx));
				}
			}
		}
	}
	CPUMatrix* multiplyA(AbstractMatrix* B) override {
		CPUMatrix* C = new CPUMatrix(x, B->x);
		multiplyA(B, C);
		return C;
	}
	void multiplyA(AbstractMatrix* B, AbstractMatrix* C) override {
		for (size_t Ax = 0; Ax < x; Ax++) {
			for (size_t Bx = 0; Bx < B->x; Bx++) {
				C->setIndex(Ax, Bx, index(0, Ax) * B->index(0, Bx));
				for (size_t Ay = 1; Ay < y; Ay++) {
					C->addIndex(Ax, Bx, index(Ay, Ax) * B->index(Ay, Bx));
				}
			}
		}
	}
	CPUMatrix* multiplyB(AbstractMatrix* B) override {
		CPUMatrix* C = new CPUMatrix(y, B->y);
		multiplyB(B, C);
		return C;
	}
	void multiplyB(AbstractMatrix* B, AbstractMatrix* C) override {
		for (size_t Ay = 0; Ay < y; Ay++) {
			for (size_t By = 0; By < B->y; By++) {
				C->setIndex(Ay, By, index(Ay, 0) * B->index(By, 0));
				for (size_t Ax = 1; Ax < x; Ax++) {
					C->addIndex(Ay, By, index(Ay, Ax) * B->index(By, Ax));
				}
			}
		}
	}
	CPUMatrix* multiplyAB(AbstractMatrix* B) override {
		CPUMatrix* C = new CPUMatrix(x, B->y);
		multiplyAB(B, C);
		return C;
	}
	void multiplyAB(AbstractMatrix* B, AbstractMatrix* C) override {
		for (size_t Ax = 0; Ax < x; Ax++) {
			for (size_t By = 0; By < B->y; By++) {
				C->setIndex(Ax, By, index(0, Ax) * B->index(By, 0));
				for (size_t Ay = 1; Ay < y; Ay++) {
					C->addIndex(Ax,By, index(Ay, Ax)*B->index(By, Ay));
				}
			}
		}
	}

	CPUMatrix* multiplyElementWise(AbstractMatrix* B) override {
		CPUMatrix* C = new CPUMatrix(y, x);
		multiplyElementWise(B, C);
		return C;
	 }
	void multiplyElementWise(AbstractMatrix* B, AbstractMatrix* C) override {
		for (size_t i = 0; i < size; i++) {
			C->setIndex(i, index(i) * B->index(i));
		}
	}

	CPUMatrix* divideElementWise(AbstractMatrix* B) override {
		CPUMatrix* C = new CPUMatrix(y, x);
		divideElementWise(B, C);
		return C;
	}
	void divideElementWise(AbstractMatrix* B, AbstractMatrix* C) override {
		for (size_t i = 0; i < size; i++) {
			C->setIndex(i, index(i) / B->index(i));
		}
	}

	CPUMatrix* sigmoid() override {
		CPUMatrix* C = new CPUMatrix(y, x);
		sigmoid(C);
		return C;
	}
	void sigmoid(AbstractMatrix* C) override {
		for (size_t i = 0; i < size; i++) {
			C->setIndex(i, 1.0 / (1.0 + std::exp(-index(i))) );
		}
	}

	CPUMatrix* sigmoidDifferential() override {
		CPUMatrix* C = new CPUMatrix(y, x);
		sigmoidDifferential(C);
		return C;
	}

	void sigmoidDifferential(AbstractMatrix* C) override {
		for (size_t i = 0; i < size; i++) {
			double x = index(i);
			C->setIndex(i, x * (1 - x));
			
		}
	}

	CPUMatrix* add(AbstractMatrix* B) override {
		CPUMatrix* C = new CPUMatrix(y, x);
		add(B, C);
		return C;
	}
	void add(AbstractMatrix* B, AbstractMatrix* C) override {
		for (size_t i = 0; i < size; i++) {
			C->setIndex(i, index(i) + B->index(i));
		}
	}

	void addAssign(AbstractMatrix* B) override {
		for (size_t i = 0; i < size; i++) {
			addIndex(i, B->index(i));
		}
	}
	
	CPUMatrix* subtract(AbstractMatrix* B) override {
		CPUMatrix* C = new CPUMatrix(y, x);
		subtract(B, C);
		return C;
	}
	void subtract(AbstractMatrix* B, AbstractMatrix* C) override {
		for (size_t i = 0; i < size; i++) {
			C->setIndex(i, index(i) - B->index(i));
		}
	}

	void subtractAssign(AbstractMatrix* B) override {
		for (size_t i = 0; i < size; i++) {
			addIndex(i, -B->index(i));
		}
	}

	CPUMatrix* addConst(double B) override {
		CPUMatrix* C = new CPUMatrix(y, x);
		addConst(B, C);
		return C;
	}
	void addConst(double B, AbstractMatrix* C) override {
		for (size_t i = 0; i < size; i++) {
			C->setIndex(i, index(i) + B);
		}
	}

	CPUMatrix* scale(double B) override {
		CPUMatrix* C = new CPUMatrix(y, x);
		scale(B, C);
		return C;
	}
	void scale(double B, AbstractMatrix* C) override {
		for (size_t i = 0; i < size; i++) {
			C->setIndex(i, index(i) * B);
		}
	}

	double sum() const override{
		double total = 0;
		for (size_t i = 0; i < size; i++) {
			total += index(i);
		}
		return total;
	}

	void randomFill(double min, double mAx) override;

	void randomFill(double lowerMin, double lowerMAx, double upperMin, double upperMAx) override;

	void fill(double value) override {
		for (size_t i = 0; i < size; i++) {
			arr[i] = value;
		}
	}

	CPUMatrix* copy_keeping_array() const override {
		return new CPUMatrix(y,x,arr,CPU);
	}

	// /**
	//  * @return a copy of the matrix on the same device
	//  */
	// CPUMatrix* copy() const override {
	// 	CPUMatrix* m = new CPUMatrix(y, x);
	// 	memcpy(m->arr, arr, size * sizeof(double));
	// 	return m;
	// }
	// /**
	//  * @param a a copy of the matrix on the same device
	//  */
	// void copy(AbstractMatrix* m) const override {
	// 	if(dynamic_cast<CPUMatrix*>(m) == NULL){
	// 		ilog(ERROR, "Cannot copy NULL or CPUMatrix to non-CPUMatrix");
	// 	}
	// 	m->x = x;
	// 	m->y = y;
	// 	memcpy(m->arr, arr, size * sizeof(double));
	// }
	private:
	void transpose(CPUMatrix* m){
		for (size_t Y = 0; Y < y; Y++) {
			for (size_t X = 0; X < x; X++) {
				setIndex(Y, X, m->index(X, Y));
			}
		}
	}
	public:
	
	void transpose() override {
		CPUMatrix m(copy());
		x = m.y;
		y = m.x;
		transpose(&m);
	}
	 CPUMatrix* transposeNew() override {
		CPUMatrix* m = new CPUMatrix(x, y);
		m->transpose(this);
		return m;
	}


	void print() const override {
		for (size_t c = 0; c < y; c++) {
			std::cout << index(c, 0);
			for (size_t r = 1; r < x; r++) {
				std::cout << ",		" << index(c, r);
			}
			std::cout << std::endl;
		}
	}

	void print(int resolution) const override {
		std::streamsize ss = std::cout.precision();
		std::cout.precision(resolution);
		for (int c = 0; c < y; c++) {
			std::cout << std::fixed << index(c, 0);
			for (int r = 1; r < x; r++) {
				std::cout << ",		" << std::fixed << index(c, r);
			}
			std::cout << std::endl;
		}
		std::cout.precision(ss);
	}



	void deconstruct() override {
		if (arr) {
			//free(arr);
			std::free(arr);
			arr = nullptr;
		}
	}

	/**
	 * used to de-refence array to prevent freeing during deconstruction
	 */
	void delete_array_ref() override {
		arr = nullptr;
	}






	void convolute(AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* out, int outY, int outX, int outZ, int convY, int convX, int convZ) override;


	/**
 * this - output error to back propigate
 * @param input matrix y: Y*Z, x: X
 * @param layer convolution matrix y: convY*Z, x: convX1 + convX2 + convX3... convX(convZ) - the Z dimension are stored adjacently in the Y axis, The convZ dimension are split into chunks in the X axis
 * @param layer_deltas the error in this conv layer (LR already applied)
 * @param bias size = convZ
 * @param prevError error at the input to the layer
 * @param out the output of the network
 * @param out_error error at the output of this layer
 * @param gradient, the gradient at the output of this layer
 * @param LR learning rate scalar to apple
 * @param outY the Y size of the output matrix = inY - floor(convY/2)-1
 * @param outX the X size of the output matrix = inX - floor(convX/2)-1
 * @param outZ the Z depth of the ouput eqault to the number of conv filters, also called f
 * @param convX the X dimension of the convolution layer
 * @param convY the Y dimension of the convolution layer
 * @param convZ the Z depth of the convolution layer, equal to the Z dimension of the input (the Z dimension of the input can be used as RGB or whatever)
 */
void convBackprop(AbstractMatrix* input, AbstractMatrix* layer, AbstractMatrix* layer_deltas, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* out, AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) override;


	private:
		inline double tanh(double x) {
			double ex = exp(x);
			double nex = exp(-x);
			auto temp = (ex - nex) / (nex + ex);
			if(isnan(temp)){
				std::cout << "E\n";
			}
			return temp;
		}
		/* inline double tanhd(double x) {
			double ex = exp(x);
			double nex = exp(-x);
			double temp = ex + nex;
			return  4 / (temp * temp);
		} */

		//Mathematically varified
		inline double tanhd_on_tanh(double x){
			return 1 - x*x;
		}
		

};