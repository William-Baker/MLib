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
#include "GPUMatrix.hpp"



class CPUMatrix : public AbstractMatrix<double> {
public:
	CPUMatrix() { //Null constructor
		arr = nullptr;
	}
	CPUMatrix(size_t Y, size_t X) {//standard constructor taking the number of columns in the matrix and the number of rows
			//then allocating the appropriate amount of memory on the device
		size = X * Y;
		arr = static_cast<double*>(malloc(size * sizeof(double)));
		y = Y;
		x = X;
	}
	/**
	 * Construct CPU matrix, using existing array
	 * Data will NOT be copied
	 * @param Y Y dimension
	 * @param X X dimension
	 * @param arr existing array to use, assumed of size Y*X
	 */
	CPUMatrix(size_t Y, size_t X, double* arr) {
		size = X * Y;
		this->arr = arr;
		y = Y;
		x = X;
	}

	double index(size_t Y, size_t X) override {
		return arr[getIndex(Y, X)];
	}
	double index(size_t i) override {
		return arr[i];
	}

	void setIndex(size_t Y, size_t X, double value) override {
		arr[getIndex(Y, X)] = value;
	}
	void setIndex(size_t i, double value) override {
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

	void randomFill(double min, double mAx) override;

	void randomFill(double lowerMin, double lowerMAx, double upperMin, double upperMAx) override;

	void fill(double value) override {
		for (size_t i = 0; i < size; i++) {
			arr[i] = value;
		}
	}

	/**
	 * @return a copy of the matrix on the same device
	 */
	CPUMatrix* copy() override {
		CPUMatrix* m = new CPUMatrix(y, x);
		memcpy(m->arr, arr, size * sizeof(double));
		return m;
	}
	/**
	 * @param a a copy of the matrix on the same device
	 */
	void copy(AbstractMatrix* m)override {
		memcpy(m->arr, arr, size * sizeof(double));
	}
	
	void transpose() override {
		auto m = copy();
		x = m->y;
		y = m->y;
		for (size_t Y = 0; Y < y; Y++) {
			for (size_t X = 0; X < x; X++) {
				setIndex(X, Y, m->getIndex(Y, X));
			}
		}
		delete(m);
	}
	 CPUMatrix* transposeNew() override {
		CPUMatrix* m = new CPUMatrix(x, y);
		this->transpose(m);
		return m;
	}


	void print() override {
		for (int c = 0; c < y; c++) {
			std::cout << index(c, 0);
			for (int r = 1; r < x; r++) {
				std::cout << ",		" << index(c, r);
			}
			std::cout << std::endl;
		}
	}

	void print(int resolution) {
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

	/**
	 * @return a copy of the matrix stored in CPU memory
	 */
	double* get_CPU_pointer(){
		double* ptr = (double*)malloc(size * sizeof(double));
		while (ptr == 0)
		{
			ptr = (double*)malloc(size * sizeof(double));
			std::cout << "Out of CPU memory" << std::endl;
		}
		memcpy(ptr, arr, size * sizeof(double));


		return ptr;
	}
	/**
	 * @return the matrix stored in CPU memory - warning this may be a direct refernce to the Matrices array
	 */
	double* get_CPU_pointer_read_only(){
		return arr;
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






	void convolute(AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* net, AbstractMatrix* out, int inX, int inY, int inZ, int outX, int outY, int outZ, int convX, int convY) override {
		for (int filter = 0; filter < outZ; filter++) {
			for (int fX = 0; fX < outX; fX++) {
				for (int fY = 0; fY < outY; fY++) {
					double temp = 0;
					for (int cX = 0; cX < convX; cX++) {
						for (int cY = 0; cY < convY * inZ; cY++) {
								temp += index(fY*inZ + cY, fX + cX) * layer->index(cY, cX + filter * convX);
						}
					}
					temp += bias->index(filter);
					net->setIndex(fY * outZ + filter, fX, temp);
					out->setIndex(fY * outZ + filter , fX, tanh(temp));
				}
			}
		}
	}


	void convBackprop(AbstractMatrix* in, AbstractMatrix* layer, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* net,  AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) override;



	private:
		void transpose(CPUMatrix* B) {
			for (size_t Y = 0; Y < y; Y++) {
				for (size_t X = 0; X < x; X++) {
					B->setIndex(X, Y, index(Y,X));
				}
			}
		}
		inline double tanh(double x) {
			double ex = exp(x);
			double nex = exp(-x);
			return (ex - nex) / (nex + ex);
		}
		inline double tanhd(double x) {
			double ex = exp(x);
			double nex = exp(-x);
			double temp = ex + nex;
			return  4 / (temp * temp);
		}
		

};

class Matrix {
private:
	static bool checked;
	static bool pHasGPU;
	AbstractMatrix<double>* m = 0;
	static bool checkGPU();
	static bool GPUStats();
public:
	static std::mt19937 mt;
	Matrix() {
		//defaultConstructor
	}
	Matrix(size_t Y, size_t X) {
		
		if (checkGPU()) {
			m = new GPUMatrix(Y, X);
		}
		else {
			m = new CPUMatrix(Y, X);
		}
	}
	Matrix(AbstractMatrix<double>* a) {
		m = a;
	}

	/**
	 * Copy any matrix to this instace, a new instance will aways be created
	 * this - the destination matrix
	 * @param A the source matrix
	 */
	void copyToThis(Matrix& A) {
		AbstractMatrix<double>* a = A.getStrategy();
		if(dynamic_cast<CPUMatrix*>(a)){
			if (checkGPU()) {	
				m = new GPUMatrix(a->y, a->x, a->arr, GPUMatrix::MEM::CPU);
			}
			else {
				m = a->copy();
			}
		}
		else if (dynamic_cast<GPUMatrix*>(a)){
			if (checkGPU()) {		
				m = a->copy();
			}
			else {
				m = new CPUMatrix(a->y, a->x, A.get_CPU_pointer());
			}
		}
	}

	Matrix(size_t Y, size_t X, double* arr){
		CPUMatrix* temp = new CPUMatrix(Y, X, arr);
		if (checkGPU()) {
			Matrix temp_wrapper(temp);
			copyToThis(temp_wrapper);
		}
		else{
			m = temp;
		}

	}
	
	template<typename T>
	void copyToThis(T* arr){
		CPUMatrix* temp = new CPUMatrix(m->y, m->x);
		for(size_t index = 0; index < temp->size; index++){
			temp->arr[index] = (double)(arr[index]);
		}
		Matrix temp_wrapper(temp);
		copyToThis(temp_wrapper);
		temp->delete_array_ref();

	}


	//TODO have a look
	/* template<typename T>
	Matrix(T* arr){
		CPUMatrix* temp = new CPUMatrix(m->y, m->x);
		for(size_t index = 0; index < temp->size; index++){
			temp->arr[index] = (double)(arr[index]);
		}
		Matrix temp_wrapper(temp);
		copyToThis(temp_wrapper);
		temp->delete_array_ref();

	} */

	int width() { return m->x; }
	int height() { return m->y; }
	int size() { return m->size; }

	double index(size_t Y, size_t X) { return m->index(Y, X); }
	double index(size_t i) { return m->index(i); }

	void setIndex(size_t Y, size_t X, double val) { m->setIndex(Y, X, val); }
	void setIndex(size_t i, double val) { 
		m->setIndex(i, val); 
	}

	void addIndex(size_t Y, size_t X, double value) { m->addIndex(Y, X, value); }
	void addIndex(size_t i, double value) { m->addIndex(i, value); }

	Matrix multiply(Matrix& B) { 
		return Matrix(m->multiply(B.getStrategy()));
	}
	void multiply(Matrix& B, Matrix& C) {
		if (width() != B.height()) {
			throw("Dimension mismatch");
		}
		m->multiply(B.getStrategy(), C.getStrategy()); 
	}

	Matrix multiplyA(Matrix& B) { return Matrix(m->multiplyA(B.getStrategy()));  }
	void multiplyA(Matrix& B, Matrix& C) { 
		if (height() != B.height()) {
			throw("Dimension mismatch");
		}
		return m->multiplyA(B.getStrategy(), C.getStrategy());  
	}

	Matrix multiplyB(Matrix& B) { return Matrix(m->multiplyB(B.getStrategy())); }
	void multiplyB(Matrix& B, Matrix& C) { 
		if (width() != B.width()) {
			throw("Dimension mismatch");
		}
		return m->multiplyB(B.getStrategy(), C.getStrategy()); 
	}

	Matrix multiplyAB(Matrix& B) { return Matrix(m->multiplyAB(B.getStrategy())); }
	void multiplyAB(Matrix& B, Matrix& C) {
		if (height() != B.width()) {
			throw("Dimension mismatch");
		}
		return m->multiplyAB(B.getStrategy(), C.getStrategy());
	}

	Matrix multiplyElementWise(Matrix& B) {
		if (width() != B.width() || height() != B.height() ) 
			throw("Dimension mismatch");
		
		
		return Matrix(m->multiplyElementWise(B.getStrategy()));
	}

	void multiplyElementWise(Matrix& B, Matrix& C){
		/* if (width() != B.width() || height() != B.height()) {
			throw("Dimension mismatch");
		} */
		if(!((width() == B.width() && width() == C.width()) && (height() == B.height() && height() == C.height()) ))
			throw("Dimension mismatch");
		 //TODO migrate this to the others
		m->multiplyElementWise(B.getStrategy(), C.getStrategy());
	}

	Matrix divideElementWise(Matrix& B) {
		if (width() != B.width() || height() != B.height()) {
			throw("Dimension mismatch");
		}
		return Matrix(m->divideElementWise(B.getStrategy()));
	}
	void divideElementWise(Matrix& B, Matrix& C) {
		if (width() != B.width() || height() != B.height()) {
			throw("Dimension mismatch");
		}
		return m->divideElementWise(B.getStrategy(), C.getStrategy());
	}

	Matrix sigmoid() {
		return Matrix(m->sigmoid());
	}
	void sigmoid(Matrix& C) {
		if (width() != C.width() || height() != C.height()) {
			throw("Dimension mismatch");
		}
		m->sigmoid(C.getStrategy());
	}

	Matrix sigmoidDifferential() {
		return Matrix(m->sigmoidDifferential());
	}
	void sigmoidDifferential(Matrix& C) {
		if (width() != C.width() || height() != C.height()) {
			throw("Dimension mismatch");
		}
		m->sigmoidDifferential(C.getStrategy());
	}

	Matrix add(Matrix& B){
		if (width() != B.width() || height() != B.height()) {
			throw("Dimension mismatch");
		}
		return Matrix(m->add(B.getStrategy()));
	}
	void add(Matrix& B, Matrix& C) {
		if (width() != B.width() || height() != B.height() ||  B.width() != C.width() || B.height() != C.height()) {
			throw("Dimension mismatch");
		}
		m->add(B.getStrategy(), C.getStrategy());
	}

	void addAssign(Matrix& B) {
		if (width() != B.width() || height() != B.height()) {
			throw("Dimension mismatch");
		}
		m->addAssign(B.getStrategy());
	}
	

	Matrix subtract(Matrix& B){
		if (width() != B.width() || height() != B.height())  throw("Dimension mismatch");
		return Matrix(m->subtract(B.getStrategy()));
	}

	void subtract(Matrix& B, Matrix& C) {
		if (width() != B.width() || height() != B.height() || B.width() != C.width() || B.height() != C.height()) {
			throw("Dimension mismatch");
		}
		m->subtract(B.getStrategy(), C.getStrategy());
	}

	void subtractAssign(Matrix& B) {
		if (width() != B.width() || height() != B.height()) {
			throw("Dimension mismatch");
		}
		m->subtractAssign(B.getStrategy());
	}

	Matrix addConst(double B){
		return Matrix(m->addConst(B));
	}
	void addConst(double B, Matrix& C){
		if (width() != C.width() || height() != C.height()) {
			throw("Dimension mismatch");
		}
		m->addConst(B, C.getStrategy());
	}

	Matrix scale(double B) {
		return *(new Matrix(m->scale(B)));
	}

	void scale(double B, Matrix& C) {
		if (width() != C.width() || height() != C.height()) {
			throw("Dimension mismatch");
		}
		m->scale(B, C.getStrategy());
	}

	void convolute(Matrix &layer, Matrix &bias, Matrix& net, Matrix &out, int inX, int inY, int inZ, int outX, int outY, int outZ, int convX, int convY) {
		m->convolute(layer.m, bias.m, net.m, out.m, inX,  inY,  inZ,  outX,  outY,  outZ,  convX,  convY);
	}
	void convBackprop(Matrix& in, Matrix& layer, Matrix& prevError, Matrix& bias, Matrix& net, Matrix& gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) {
		m->convBackprop(in.m, layer.m, prevError.m, bias.m, net.m, gradient.m, outY, outX, outZ, convY, convX, convZ, LR);
	}

	void randomFill(double min, double max) { m->randomFill(min, max); }
	void randomFill(double negmin, double negmax, double min, double max) { m->randomFill(negmin, negmax, min, max); }

	void fill(double value) {
		m->fill(value);
	}

	void transpose() { m->transpose(); }
	Matrix transposeNew() { return Matrix(m->transposeNew()); }

	/**
	 * @return a copy of the matrix on the same device
	 */
	Matrix copy() { return Matrix(m->copy()); }
	/**
	 * @param a a copy of the matrix on the same device
	 */
	void copy(Matrix& a) {m->copy(a.getStrategy());}




	

	void print() { m->print(); }
	void print(int resolution) { m->print(resolution); }



	AbstractMatrix<double>* getStrategy() { return m; }

	static void forceUseGPU();
	static void forceUseCPU();
	static bool hasGPU() {
		return checkGPU();
	}

	double* get_CPU_pointer() { return m->get_CPU_pointer(); }

	double* get_CPU_pointer_read_only() { return m->get_CPU_pointer_read_only(); }
	

	static std::string compare(Matrix& a, Matrix& b, double expectedMin, double expectedMAx) {
		double* A = a.get_CPU_pointer_read_only();
		double* B = b.get_CPU_pointer_read_only();
		size_t size = a.size();
		bool dissimilar = false;
		int errorA = 0;
		int errorB = 0;
		double error = 0;
		std::string dissimilar_indeces;//Keep track of all the dissimilar indeces for debugging
		for (long i = 0; i < size; i++) {
			if (A[i] != B[i]) {
				dissimilar = true;
				dissimilar_indeces += std::to_string(i) + ", ";
				error += abs(A[i] - B[i]);
				if (A[i] < expectedMin || A[i] > expectedMAx) {
					errorA++;
				}
				if (B[i] < expectedMin || B[i]>expectedMAx) {
					errorB++;
				}
			}
		}
		std::string ret = "";
		if (dissimilar) {
			std::ostringstream streamObj;
			//streamObj << std::fixed;
			streamObj << error / (double)size;
			ret += "dissimilar: " + streamObj.str();
			//ret += "\nIndeces: ";
			//ret+= dissimilar_indeces;
		}
		if (errorA) {
			ret += "--A out of bounds: " + std::to_string(errorA) + " times--";
		}
		if (errorB) {
			ret += "--B out of bounds: " + std::to_string(errorB) + " times--";	
		}
		if (ret.length() > 0)  ret += "\n";
		return ret;
	}

	/**
	 * used to de-refence array to prevent freeing during deconstruction
	 */
	void delete_array_ref(){
		if(m) {
			m->delete_array_ref();
		}
	}

	void delete_implementation(){
		m = 0;
	}

	~Matrix() {
		if (m) {
			m->deconstruct();
		}
	}



	Matrix& operator=(Matrix&& B) {
		m = (std::move(B.m));
		B.delete_implementation();
		return *this;
	}
	
	Matrix(const Matrix& B ){
		m = (std::move(B.m));
	}
};

class Timer {
	static std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
public:
	static void start();
	static size_t time();
};


