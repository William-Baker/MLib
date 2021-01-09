
#pragma once
#include <cstddef>
template < typename E> class MLStruct {
public:
	enum StructType{
		STRUCT_GPUMatrix,
		STRUCT_CPUMatrix,
		STRUCT_CPUTensor,
		STRUCT_GPUTensor,
		STRUCT_UNKNOWN
	};

	StructType struct_type = STRUCT_UNKNOWN;
	
	virtual void print() const = 0;
	int get_size() const {return size;}
	virtual E* get_implementation_array() = 0;
	virtual const E* get_implementation_array() const = 0;
protected:
	size_t size = 0;
};

// Template class for Matrices
// Assumes storeage in Column major format
// 
template < typename E> class AbstractMatrix : public MLStruct<E> {
public:
	E* arr = 0;
	size_t x = 0;
	size_t y = 0;

		

	virtual void deconstruct() = 0;
	virtual void delete_array_ref() = 0;


	E* get_implementation_array() override { return arr; }
	const E* get_implementation_array() const override { return arr; }
	/**
	 * @return a copy of the matrix stored in host memory
	 */
	virtual E* copy_array_host() const = 0;
	/**
	 * @return the matrix stored in source host memory - warning this may be a direct refernce to the Matrices array
	 */
	virtual const E* get_array_host() const = 0;



	virtual E index(size_t Y, size_t X) const = 0; //const indexing so consistent for GPU
	virtual E index(size_t i) const = 0;

	static inline size_t indexing(size_t Y, size_t X, size_t y_dim) {
		return X*y_dim + Y;
	}
	static inline size_t indexing_t(size_t Y, size_t X, size_t x_dim){
		return Y*x_dim + X;
	}

	inline size_t getIndex(size_t Y, size_t X) const {
		return indexing(Y, X, y);
	}
	inline size_t getIndex_t(size_t Y, size_t X) const {
		return indexing_t(Y, X, x);
	}

	virtual void setIndex(size_t Y, size_t X, E value) = 0;
	virtual void setIndex(size_t i, E value) = 0;

	virtual void randomFill(E min, E max) = 0;
	virtual void randomFill(E lowerMin, E lowerMax, E upperMin, E upperMax) = 0;

	virtual void fill(E value) = 0;

	virtual void transpose() = 0;
	virtual AbstractMatrix* transposeNew() = 0;

	//Standard multiply A and B
	virtual AbstractMatrix* multiply(AbstractMatrix* B) = 0;
	virtual void multiply(AbstractMatrix* B, AbstractMatrix* C) = 0;

	//Standard multiply A_t and B
	virtual AbstractMatrix* multiplyA(AbstractMatrix* B) = 0;
	virtual void multiplyA(AbstractMatrix* B, AbstractMatrix* C) = 0;

	//Standard multiply A and B_t
	virtual AbstractMatrix* multiplyB(AbstractMatrix* B) = 0;
	virtual void multiplyB(AbstractMatrix* B, AbstractMatrix* C) = 0;

	//Standard multiply A_t and B_t
	virtual AbstractMatrix* multiplyAB(AbstractMatrix* B) = 0;
	virtual void multiplyAB(AbstractMatrix* B, AbstractMatrix* C) = 0;

	virtual AbstractMatrix* multiplyElementWise(AbstractMatrix* B) = 0;
	virtual void multiplyElementWise(AbstractMatrix* B, AbstractMatrix* C) = 0;

	virtual AbstractMatrix* divideElementWise(AbstractMatrix* B) = 0;
	virtual void divideElementWise(AbstractMatrix* B, AbstractMatrix* C) = 0;

	virtual AbstractMatrix* sigmoid() = 0;
	virtual void sigmoid(AbstractMatrix* C) = 0;
	
	virtual AbstractMatrix* sigmoidDifferential() = 0;
	virtual void sigmoidDifferential(AbstractMatrix* C) = 0;

	virtual AbstractMatrix* add(AbstractMatrix* B) = 0;
	virtual void add(AbstractMatrix* B, AbstractMatrix* C) = 0;
	
	virtual void addAssign(AbstractMatrix* B) = 0;

	virtual AbstractMatrix* subtract(AbstractMatrix* B) = 0;
	virtual void subtract(AbstractMatrix* B, AbstractMatrix* C) = 0;

	virtual void subtractAssign(AbstractMatrix* B) = 0;

	virtual AbstractMatrix* addConst(double B) = 0;
	virtual void addConst(double B, AbstractMatrix* C) = 0;

	virtual AbstractMatrix* scale(double B) = 0;
	virtual void scale(double B, AbstractMatrix* C) = 0;

	virtual double sum() const = 0;

	/**
	 * used when wanting to have two matrices pointing to the same data with different dimensions
	 * @returns a new abstarct matrix pointing to the same array as this
	 */
	virtual AbstractMatrix* copy_keeping_array() const = 0;

	//Add at index
	virtual void addIndex(size_t Y, size_t X, double value) = 0;
	virtual void addIndex(size_t i, double value) = 0;

	virtual void print() const = 0;
	virtual void print(int resolution) const = 0;



	enum MEM { CPU, GPU };









	//Addons
	virtual void convolute(AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* out, int outY, int outX, int outZ, int convY, int convX, int convZ) = 0;
/**
 * this - output error to back propigate
 * @param input matrix y: Y*Z, x: X
 * @param layer convolution matrix y: convY*Z, x: convX1 + convX2 + convX3... convX(convZ) - the Z dimension are stored adjacently in the Y axis, The convZ dimension are split into chunks in the X axis
 * @param this_layer_conv_error the error in this conv layer (LR already applied)
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
	virtual void convBackprop(AbstractMatrix* input, AbstractMatrix* layer, AbstractMatrix* this_layer_conv_error, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* out, AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) = 0;

};
