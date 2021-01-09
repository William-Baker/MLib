#pragma once
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <initializer_list>
#include "IO.hpp"
#include "GPUMatrix.hpp"
#include "CPUMatrix.hpp"


#define CHECK_DIMS //enables dimension checking for all mathematical matrix calls


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
	Matrix(size_t Y, size_t X) : Matrix() {
		
		if (checkGPU()) {
			m = new GPUMatrix(Y, X);
		}
		else {
			m = new CPUMatrix(Y, X);
		}
	}
	Matrix(AbstractMatrix<double>* a) : Matrix() {
		m = a;
	}

	/**
	 * @param s structure to convert to a matrix
	 * @param deconstruct_implementation true to delete the original when deconstructing, set to false if using a matrix as a wrapper
	 */
	Matrix(MLStruct<double>* s, bool deconstruct_implementation) : Matrix(){
		this->deconstruct_implementation = deconstruct_implementation;
		if(s->struct_type == MLStruct<double>::StructType::STRUCT_GPUMatrix){
			m = static_cast<AbstractMatrix<double>*>(s);
		}
		else if(s->struct_type == MLStruct<double>::StructType::STRUCT_CPUMatrix){
			m = static_cast<AbstractMatrix<double>*>(s);
		}
		else if(s->struct_type == MLStruct<double>::StructType::STRUCT_UNKNOWN){
			ilog(FATAL_ERROR, "Bad struct type, you forgot to set the structure type when constructing");
		}
		else{
			ilog(FATAL_ERROR, "unrecognisedstruct type");
		}
		
	}

	Matrix(const Matrix& B ) = delete;

	Matrix(Matrix && B ) : Matrix(){
		m = B.m;
		B.m = 0;
	}

	Matrix(Matrix& B ) = delete;

	// /**
	//  * Copy any matrix to this instace, a new instance will aways be created
	//  * this - the destination matrix
	//  * @param A the source matrix
	//  */
	// void copyToThis(const Matrix& A) {
	// 	const AbstractMatrix<double>* a = A.getStrategy();
	// 	if(dynamic_cast<const CPUMatrix*>(a)){
	// 		const CPUMatrix* actual = static_cast<const CPUMatrix*>(a);
	// 		if (checkGPU()) {	
	// 			m = new GPUMatrix(actual->copy());
	// 		}
	// 		else {
	// 			m = a->copy();
	// 		}
	// 	}
	// 	else if (dynamic_cast<const GPUMatrix*>(a)){
	// 		if (checkGPU()) {		
	// 			m = a->copy();
	// 		}
	// 		else {
	// 			m = new CPUMatrix(a->y, a->x, A.copy_to_CPU());
	// 		}
	// 	}
	// }


	Matrix(size_t Y, size_t X, double* arr) : Matrix(){
		CPUMatrix* temp = new CPUMatrix(Y, X, arr);
		if (checkGPU()) {
			m = new GPUMatrix(temp);
			delete temp;
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
	
	private:
	template<typename T>
	void template_construct_host_array(size_t Y, size_t X, T* arr){
		CPUMatrix* temp = new CPUMatrix(m->y, m->x);
		for(size_t index = 0; index < temp->get_size(); index++){
			temp->arr[index] = (double)(arr[index]);
		}
		if(checkGPU()){
			m = new GPUMatrix(temp);
			delete temp;
		}
		else
		{
			m = temp;
		}
	}
	public:

	template<typename T>
	Matrix(size_t Y, size_t X, T* arr) : Matrix(){
		template_construct_host_array(Y, X, arr);
	}

	template<typename T>
	void copyToThis(T* arrIn){
		template_construct_host_array(m->x, m->y, arrIn);
	}

	/**
	 * Copy a host array to this matrix
	 */
	
	void copyToThis(double* arr){
		CPUMatrix* temp = new CPUMatrix(m->y, m->x, arr);
		
		delete m; //TODO might have to change to if m delte m

		if(checkGPU()) m = new GPUMatrix(temp);
		else m = new CPUMatrix(temp);
	}

	Matrix copy_keeping_same_data(){
		return Matrix(m->copy_keeping_array());
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
	Matrix copy() const {
		if(checkGPU()){ //we have a GPU
			return Matrix(new GPUMatrix(m));

		}
		else{
			return Matrix(new CPUMatrix(m));
			
		}
	}


	AbstractMatrix<double>* getStrategy() { return m; }
	const AbstractMatrix<double>* getStrategy() const { return m; }

	static void forceUseGPU();
	static void forceUseCPU();
	static bool checkGPU();
	static void resetGPUState();
	static bool usingGPU();


	double* get_implementation_array() { return m->get_implementation_array(); }
	const double* get_implementation_array() const { return m->get_implementation_array(); }


	/**
	 * @return a copy of the matrix stored in host memory
	 */
	double* copy_array_host() const {return m->copy_array_host();}
	/**
	 * @return the matrix stored in source (device/host) memory - warning this may be a direct refernce to the Matrices array
	 */
	const double* get_array_host() const {return m->get_array_host();}
	
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
			ilog(FATAL_ERROR, "bad index");
		}
		return m->index(Y, X); 
	}
	double index(size_t i) {
		if(i >= size()){
			ilog(FATAL_ERROR, "bad index");
		}
		return m->index(i); 
	}

	void setIndex(size_t Y, size_t X, double val) {
		if(Y >= height() || X >= width()){
			ilog(FATAL_ERROR, "bad index");
		}
		m->setIndex(Y, X, val);
	}
	void setIndex(size_t i, double val) {
		if(i >= size()){
			ilog(FATAL_ERROR, "bad index");
		}
		m->setIndex(i, val); 
	}

	void addIndex(size_t Y, size_t X, double value) {
		if(Y >= height() || X >= width()){
			ilog(FATAL_ERROR, "bad index");
		}
		m->addIndex(Y, X, value); 
	}
	void addIndex(size_t i, double value) {
		if(i >= size()){
			ilog(FATAL_ERROR, "bad index");
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
		#ifdef CHECK_DIMS
		if (width() != B.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		#endif
		m->multiply(B.getStrategy(), C.getStrategy()); 
	}

	Matrix multiplyA(Matrix& B) { return Matrix(m->multiplyA(B.getStrategy()));  }
	void multiplyA(Matrix& B, Matrix& C) { 
		#ifdef CHECK_DIMS
		if (height() != B.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		#endif
		return m->multiplyA(B.getStrategy(), C.getStrategy());  
	}

	Matrix multiplyB(Matrix& B) { return Matrix(m->multiplyB(B.getStrategy())); }
	void multiplyB(Matrix& B, Matrix& C) { 
		#ifdef CHECK_DIMS
		if (width() != B.width()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		#endif
		return m->multiplyB(B.getStrategy(), C.getStrategy()); 
	}

	Matrix multiplyAB(Matrix& B) { return Matrix(m->multiplyAB(B.getStrategy())); }
	void multiplyAB(Matrix& B, Matrix& C) {
		#ifdef CHECK_DIMS
		if (height() != B.width()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		#endif
		return m->multiplyAB(B.getStrategy(), C.getStrategy());
	}

	Matrix multiplyElementWise(Matrix& B) {
		#ifdef CHECK_DIMS
		if(!((width() == B.width()) && (height() == B.height()) ))
			ilog(FATAL_ERROR, "Dimension mismatch");
		#endif
		return Matrix(m->multiplyElementWise(B.getStrategy()));
	}

	void multiplyElementWise(Matrix& B, Matrix& C){
		#ifdef CHECK_DIMS
		if(!((width() == B.width() && width() == C.width()) && (height() == B.height() && height() == C.height()) ))
			ilog(FATAL_ERROR, "Dimension mismatch");
		#endif
		m->multiplyElementWise(B.getStrategy(), C.getStrategy());
	}

	Matrix divideElementWise(Matrix& B) {
		#ifdef CHECK_DIMS
		if(!((width() == B.width()) && (height() == B.height()) ))
			ilog(FATAL_ERROR, "Dimension mismatch");
		#endif
		return Matrix(m->divideElementWise(B.getStrategy()));
	}
	void divideElementWise(Matrix& B, Matrix& C) {
		#ifdef CHECK_DIMS
		if(!((width() == B.width() && width() == C.width()) && (height() == B.height() && height() == C.height()) ))
			ilog(FATAL_ERROR, "Dimension mismatch");
		#endif
		return m->divideElementWise(B.getStrategy(), C.getStrategy());
	}

	Matrix sigmoid() {
		return Matrix(m->sigmoid());
	}
	void sigmoid(Matrix& C) {
		if (width() != C.width() || height() != C.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		m->sigmoid(C.getStrategy());
	}

	Matrix sigmoidDifferential() {
		return Matrix(m->sigmoidDifferential());
	}
	void sigmoidDifferential(Matrix& C) {
		if (width() != C.width() || height() != C.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		m->sigmoidDifferential(C.getStrategy());
	}

	Matrix add(Matrix& B){
		if (width() != B.width() || height() != B.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		return Matrix(m->add(B.getStrategy()));
	}
	void add(Matrix& B, Matrix& C) {
		if (width() != B.width() || height() != B.height() ||  B.width() != C.width() || B.height() != C.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		m->add(B.getStrategy(), C.getStrategy());
	}

	void addAssign(Matrix& B) {
		if (width() != B.width() || height() != B.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		m->addAssign(B.getStrategy());
	}
	

	Matrix subtract(Matrix& B){
		if (width() != B.width() || height() != B.height())  ilog(FATAL_ERROR, "Dimension mismatch");
		return Matrix(m->subtract(B.getStrategy()));
	}

	void subtract(Matrix& B, Matrix& C) {
		if (width() != B.width() || height() != B.height() || B.width() != C.width() || B.height() != C.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		m->subtract(B.getStrategy(), C.getStrategy());
	}

	void subtractAssign(Matrix& B) {
		if (width() != B.width() || height() != B.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		m->subtractAssign(B.getStrategy());
	}

	Matrix addConst(double B){
		return Matrix(m->addConst(B));
	}
	void addConst(double B, Matrix& C){
		if (width() != C.width() || height() != C.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		m->addConst(B, C.getStrategy());
	}

	Matrix scale(double B) {
		return Matrix(m->scale(B));
	}

	void scale(double B, Matrix& C) {
		if (width() != C.width() || height() != C.height()) {
			ilog(FATAL_ERROR, "Dimension mismatch");
		}
		m->scale(B, C.getStrategy());
	}

	double sum(){
		return m->sum();
	}

	void convolute(Matrix &layer, Matrix &bias, Matrix &out,  int outY, int outX, int outZ, int convY, int convX, int convZ) {
		m->convolute(layer.m, bias.m, out.m, outY, outX, outZ, convY, convX, convZ);
	}
	/**
	 * Called iun contex of input matrix
	 */
	void convBackprop(Matrix& input, Matrix& layer, Matrix& this_layer_conv_error, Matrix& prevError, Matrix& bias, Matrix& out, Matrix& out_error, Matrix& gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) {
		if(width() != out.width() || height() != out.height()){
			ilog(FATAL_ERROR, "Dimension missmatch");
		}
		m->convBackprop(input.m, layer.m, this_layer_conv_error.m, prevError.m, bias.m, out.m, gradient.m, outY, outX, outZ, convY, convX, convZ, LR);
	}

	void randomFill(double min, double max) { m->randomFill(min, max); }
	void randomFill(double negmin, double negmax, double min, double max) { m->randomFill(negmin, negmax, min, max); }


	void flatten(){
		m->y = m->get_size();
		m->x = 1;
	}








	//-------------------------------------- Utility -------------------------------------------------------------
	static std::string compare(Matrix& a, Matrix& b, double expectedMin, double expectedMAx) {
		const double acceptable_error = 1e-5;
		
		const double* A = a.get_array_host();
		const double* B = b.get_array_host();
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

	class MaintainState{
		bool state_using_gpu;
	public:
		MaintainState() : state_using_gpu(Matrix::usingGPU()){}
		~MaintainState(){
			if(state_using_gpu) Matrix::forceUseGPU();
			else Matrix::forceUseCPU();
		}
	};

};




