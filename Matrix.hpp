#pragma once
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <initializer_list>
#include "IO.hpp"
#include "GPUMatrix.hpp"
#include "CPUMatrix.hpp"



class Matrix {
private:
	static bool checked;
	static bool hasGPU;
	AbstractMatrix<double>* m = 0;
	bool deconstruct_implementation = true;
	
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
	 * @param s structure to convert to a matrix
	 * @param deconstruct_implementation true to delete the original when deconstructing, set to false if using a matrix as a wrapper
	 */
	Matrix(MLStruct<double>* s, bool deconstruct_implementation){
		this->deconstruct_implementation = deconstruct_implementation;
		if(s->struct_type == MLStruct<double>::StructType::STRUCT_GPUMatrix){
			m = static_cast<AbstractMatrix<double>*>(s);
		}
		else if(s->struct_type == MLStruct<double>::StructType::STRUCT_CPUMatrix){
			m = static_cast<AbstractMatrix<double>*>(s);
		}
		else if(s->struct_type == MLStruct<double>::StructType::STRUCT_UNKNOWN){
			throw("Bad struct type, you forgot to set the structure type when constructing");
		}
		else{
			throw("unrecognisedstruct type");
		}
		
	}

	Matrix(const Matrix& B ) = delete;

	Matrix(Matrix && B ){
		m = B.m;
		B.m = 0;
	}

	Matrix(Matrix& B ) = delete;

	/**
	 * Copy any matrix to this instace, a new instance will aways be created
	 * this - the destination matrix
	 * @param A the source matrix
	 */
	void copyToThis(const Matrix& A) {
		const AbstractMatrix<double>* a = A.getStrategy();
		if(dynamic_cast<const CPUMatrix*>(a)){
			if (checkGPU()) {	
				m = new GPUMatrix(a->y, a->x, a->arr, GPUMatrix::MEM::CPU);
			}
			else {
				m = a->copy();
			}
		}
		else if (dynamic_cast<const GPUMatrix*>(a)){
			if (checkGPU()) {		
				m = a->copy();
			}
			else {
				m = new CPUMatrix(a->y, a->x, A.copy_to_CPU());
			}
		}
	}


	Matrix(size_t Y, size_t X, double* arr){
		CPUMatrix* temp = new CPUMatrix(Y, X, arr);
		if (checkGPU()) {
			Matrix temp_wrapper(temp);
			copyToThis(temp_wrapper);
			temp_wrapper.delete_array_ref();//we dont want to delete arr
		}
		else{
			m = temp;
		}

	}

	/* Matrix(size_t Y, size_t X, std::initializer_list<double> arr){
		CPUMatrix* temp = new CPUMatrix(Y, X, arr);
		if (checkGPU()) {
			Matrix temp_wrapper(temp);
			copyToThis(temp_wrapper);
		}
		else{
			m = temp;
		}

	} */
	
	template<typename T>
	void copyToThis(T* arr){
		CPUMatrix* temp = new CPUMatrix(m->y, m->x);
		for(size_t index = 0; index < temp->get_size(); index++){
			temp->arr[index] = static_cast<double>(arr[index]);
		}
		Matrix temp_wrapper(temp);
		copyToThis(temp_wrapper);
		temp->delete_array_ref();

	}


	//TODO have a look
	template<typename T>
	Matrix(size_t Y, size_t X, T* arr){
		CPUMatrix* temp = new CPUMatrix(m->y, m->x);
		for(size_t index = 0; index < temp->get_size(); index++){
			temp->arr[index] = (double)(arr[index]);
		}
		Matrix temp_wrapper(temp);
		copyToThis(temp_wrapper);
		temp->delete_array_ref();

	}

	Matrix& operator=(Matrix&& B) {
		m = B.m;
		B.delete_implementation();
		return *this;
	}
	


	int width() const { return m->x; }
	int height() const { return m->y; }
	int size() const { return m->get_size(); }



	/**
	 * @return a copy of the matrix on the same device
	 */
	Matrix copy() const { return Matrix(m->copy()); }
	/**
	 * @param a a copy of the matrix on the same device
	 */
	void copy(Matrix& a) const {m->copy(a.getStrategy());}




	AbstractMatrix<double>* getStrategy() { return m; }
	const AbstractMatrix<double>* getStrategy() const { return m; }

	static void forceUseGPU();
	static void forceUseCPU();
	static bool checkGPU();

	double* copy_to_CPU() const {return m->copy_to_CPU();}

	const double* get_CPU_pointer() const { return m->get_CPU_pointer(); }
	

	
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
		if (m && deconstruct_implementation) {
			m->deconstruct();
		}
	}



	

	void print() { m->print(); }
	void print(int resolution) { m->print(resolution); }

















//---------------------------------- Operations ------------------------------------------------------------

	double index(size_t Y, size_t X) {
		if(Y >= height() || X >= width()){
			throw("bad index");
		}
		return m->index(Y, X); 
	}
	double index(size_t i) {
		if(i >= size()){
			throw("bad index");
		}
		return m->index(i); 
	}

	void setIndex(size_t Y, size_t X, double val) {
		if(Y >= height() || X >= width()){
			throw("bad index");
		}
		m->setIndex(Y, X, val);
	}
	void setIndex(size_t i, double val) {
		if(i >= size()){
			throw("bad index");
		}
		m->setIndex(i, val); 
	}

	void addIndex(size_t Y, size_t X, double value) {
		if(Y >= height() || X >= width()){
			throw("bad index");
		}
		m->addIndex(Y, X, value); 
	}
	void addIndex(size_t i, double value) {
		if(i >= size()){
			throw("bad index");
		}
		m->addIndex(i, value); 
		}

	

	void fill(double value) {
		m->fill(value);
	}



	void transpose() { m->transpose(); }
	Matrix transposeNew() { return Matrix(m->transposeNew()); }


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
		return Matrix(m->scale(B));
	}

	void scale(double B, Matrix& C) {
		if (width() != C.width() || height() != C.height()) {
			throw("Dimension mismatch");
		}
		m->scale(B, C.getStrategy());
	}

	double sum(){
		return m->sum();
	}

	void convolute(Matrix &layer, Matrix &bias, Matrix &out,  int outY, int outX, int outZ, int convY, int convX, int convZ) {
		m->convolute(layer.m, bias.m, out.m, outY, outX, outZ, convY, convX, convZ);
	}
	void convBackprop(Matrix& in, Matrix& layer, Matrix& this_layer_conv_error, Matrix& prevError, Matrix& bias, Matrix& out, Matrix& out_error, Matrix& gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) {
		m->convBackprop(in.m, layer.m, this_layer_conv_error.m, prevError.m, bias.m, out.m, out_error.m , gradient.m, outY, outX, outZ, convY, convX, convZ, LR);
	}

	void randomFill(double min, double max) { m->randomFill(min, max); }
	void randomFill(double negmin, double negmax, double min, double max) { m->randomFill(negmin, negmax, min, max); }











	//-------------------------------------- Utility -------------------------------------------------------------
	static std::string compare(Matrix& a, Matrix& b, double expectedMin, double expectedMAx) {
		const double acceptable_error = 1e-5;
		
		const double* A = a.get_CPU_pointer();
		const double* B = b.get_CPU_pointer();
		size_t size = a.size();
		bool dissimilar = false;
		int errorA = 0;
		int errorB = 0;
		double error = 0;
		std::string dissimilar_indeces;//Keep track of all the dissimilar indeces for debugging
		for (long i = 0; i < size; i++) {
			if ((A[i] < (B[i] - acceptable_error)) || (A[i] > (B[i] + acceptable_error))) {
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


};




