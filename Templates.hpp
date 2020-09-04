
template < typename E> class MLStruct {
public:
	E* arr = 0;
	size_t size = 0;
	virtual void print() = 0;
};

// Template class for Matrices
// Assumes storeage in Column major format
// 
template < typename E> class AbstractMatrix : public MLStruct<E> {
public:
	
	size_t x = 0;
	size_t y = 0;

	E* getArray() { return this->arr; }

	virtual E index(size_t Y, size_t X) = 0;
	virtual E index(size_t i) = 0;

	inline size_t getIndex(size_t Y, size_t X) {
		return X*y + Y;
	}
	inline size_t getIndex_t(size_t Y, size_t X) {
		return Y * x + y;
	}
	virtual void setIndex(size_t Y, size_t X, E value) = 0;
	virtual void setIndex(size_t i, E value) = 0;

	virtual void randomFill(E min, E mAx) = 0;
	virtual void randomFill(E lowerMin, E lowerMAx, E upperMin, E upperMAx) = 0;

	virtual void fill(E value) = 0;

	virtual AbstractMatrix* copy() = 0;
	virtual void copy(AbstractMatrix* m) = 0;
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


	//Add at index
	virtual void addIndex(size_t Y, size_t X, double value) = 0;
	virtual void addIndex(size_t i, double value) = 0;

	virtual void print() = 0;
	virtual void print(int resolution) = 0;

	/**
	 * @return a copy of the matrix stored in CPU memory
	 */
	virtual E* get_CPU_pointer() = 0;
	/**
	 * @return the matrix stored in CPU memory - warning this may be a direct refernce to the Matrices array
	 */
	virtual E* get_CPU_pointer_read_only() = 0;

	virtual void deconstruct() = 0;
	virtual void delete_array_ref() = 0;
	//virtual ~AbstractMatrix() {};










	//Addons
	virtual void convolute(AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* net, AbstractMatrix* out, int inX, int inY, int inZ, int outX, int outY, int outZ, int convX, int convY) = 0;

	virtual void convBackprop(AbstractMatrix* in, AbstractMatrix* layer, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* net, AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) = 0;

};
