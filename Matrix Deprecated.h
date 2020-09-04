#pragma once

namespace MatrixGlobals {
#ifdef  GPU
	cublasHandle_t GPUHandle = NULL;
	int MaxThreads = 0;
#endif //  GPU

	bool GPUPresent = false;
	bool HasCheckedForGPU = false;
}


class Matrix {
private:
	bool HasGPU() {
		MatrixGlobals::GPUPresent = false;

#ifdef GPU
		cublasCreate(&MatrixGlobals::GPUHandle);

		//CreateCublasHandle(&MatrixGlobals::GPUHandle); //creates a handle for cuda accelerated operations to be attached to when using cuda API functions
		if (MatrixGlobals::GPUHandle == NULL) {
			MatrixGlobals::GPUPresent = false;
		}
		else {
			MatrixGlobals::GPUPresent = true;
			int nDevices;
			int ThreadCount = 0;
			cudaGetDeviceCount(&nDevices);
			for (int i = 0; i < nDevices; i++) {
				cudaDeviceProp prop;
				cudaGetDeviceProperties(&prop, i);
				int* threads = prop.maxThreadsDim;
				if (threads[0] > ThreadCount) {
					ThreadCount = threads[0];
				}
			}
			MatrixGlobals::MaxThreads = ThreadCount;
		}
#endif // GPU


		return MatrixGlobals::GPUPresent;
	}


	//----------------------------------------------------------------------    GPU Functions    ------------------------------------------------------------------------------------------

	double GPUIndex(size_t Y, size_t X) {//Gets a value from the matrix indexed in both dimensions
		double ret = 0;
		cudaMemcpy(&ret, &arrAy[Y * x + X], sizeof(double), cudaMemcpyDeviceToHost); //copy's the value from GPU memory to host memory
		return ret;
	}

	double GPUIndexVect(size_t X) {//Gets a value from the matrix indexed by a single dimension 0 to Matrix size
		double ret = 0;
		cudaMemcpy(&ret, &arrAy[X], sizeof(double), cudaMemcpyDeviceToHost); //copy's the value from GPU memory to host memory
		return ret;
	}

	void GPUSetIndex(size_t Y, size_t X, double Val) { //set the value of an co-ordinate in the GPU matrix by copying the parameter from the CPU to the GPU device at that memory location
		double temp = Val;
		cudaMemcpy(&arrAy[Y * x + X], &temp, sizeof(double), cudaMemcpyHostToDevice);
	}
	void GPUSetIndexVect(size_t X, double Val) { //set the value of an index in the GPU matrix arrAy by copying the parameter from the CPU to the GPU device at that memory location
		double temp = Val;
		cudaMemcpy(&(arrAy[X]), &temp, sizeof(double), cudaMemcpyHostToDevice);
	}

	void GPUAppendIndex(size_t Y, size_t X, double Val) { //set the value of an co-ordinate in the GPU matrix by copying the parameter from the CPU to the GPU device at that memory location
		double temp = Val + Index(Y, X);
		cudaMemcpy(&arrAy[Y * x + X], &temp, sizeof(double), cudaMemcpyHostToDevice);
	}
	void GPUAppendVect(size_t X, double Val) { //set the value of an index in the GPU matrix arrAy by copying the parameter from the CPU to the GPU device at that memory location
		double temp = Val + Index(X);
		cudaMemcpy(&(arrAy[X]), &temp, sizeof(double), cudaMemcpyHostToDevice);
	}


	void GPURandomFill(double min, double max) { //fills the matrix with a random values in the desired range
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> RandDist(min, max);
		double* TemporaryarrAy;
		TemporaryarrAy = (double*)malloc(sizeof(double) * size);
		for (int index = 0; index < size; index++) {
			TemporaryarrAy[index] = RandDist(mt);
		}
		cudaMemcpy(arrAy, TemporaryarrAy, sizeof(double) * size, cudaMemcpyHostToDevice);
		free(TemporaryarrAy);

	}
	void GPURandomFillDouble(double negmin, double negmax, double min, double max) { //fills the matrix with random values in the given ranges, used to exclude 0's from the range required for weights and biases
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> SignRandDist(0, 2);
		std::uniform_real_distribution<double> RandDist(min, max);
		double* TemporaryarrAy;
		TemporaryarrAy = (double*)malloc(sizeof(double) * size);
		for (int index = 0; index < size; index++) {
			if (SignRandDist(mt) > 1) {
				TemporaryarrAy[index] = RandDist(mt);
			}
			else {
				TemporaryarrAy[index] = 0 - RandDist(mt);
			}
		}
		cudaMemcpy(arrAy, TemporaryarrAy, sizeof(double) * size, cudaMemcpyHostToDevice);
		free(TemporaryarrAy);
	}

	void GPUFill(double Val) { //fills the matrix with a given value, needs conversion to kernel 
		cudaMemset2D(arrAy, sizeof(double), Val, x, y);
	}

	void GPUTranspose() { //transposes a GPU matrix using the CPU, needs GPU conversion
		Matrix C(x, y);
		for (int row = 0; row < x; row++) {
			for (int col = 0; col < y; col++) {
				C.SetIndex(row, col, Index(col, row));
			}
		}
		memcpy(arrAy, C.arrAy, size);
		x = C.x;
		y = C.y;
	}

	Matrix GPUTransposeReturning() { //transposes a GPU matrix using the CPU, needs GPU conversion
		Matrix C(x, y);
		for (int row = 0; row < x; row++) {
			for (int col = 0; col < y; col++) {
				C.SetIndex(row, col, Index(col, row));
			}
		}
		return C;
	}



	Matrix GPUMultiply(Matrix B) { //multiplies two matrices on the GPU 

		Matrix C(y, B.x);

		//if (x != B.y) { //check that the matrices are correct size for multiplying
		//	throw "Cannot multiply these"; //throw exception if they are
		//}
		//int Grid = (size * B.x) / MatrixGlobals::MaxThreads;//y * x * B.x;

		double scalar = 1024 / (y * B.x);
		dim3 Block(y * scalar, B.x * scalar, 1);
		dim3 Grid(ceil(y / Block.x), ceil(B.x / Block.y), 1);

		GPUMultKernel << < Grid, Block >> > (arrAy, B.arrAy, C.arrAy, y, B.x, x);

		return C;
		//bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		//if (y * B.x <= MatrixGlobals::MaxThreads) {
		//	ThreadMode = true;
		//}
		//dim3 Thread; //dimensions of the number of threads to start on the GPU
		//Thread.x = y;
		//Thread.z = 1; //1 indicating calling 1 x Grid.x
		//dim3 Block; //Dimensions of the number of blocks to be called in each thread
		//Block.y = 1;
		//Block.z = 1;
		//if (ThreadMode) {
		//	Thread.y = B.x;
		//	Block.x = 1;
		//	Kernel::matrixMultiplicationKernelThreadMode << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x);
		//}
		//else {
		//	Thread.y = 1;
		//	Block.x = B.x;
		//	Kernel::matrixMultiplicationKernel << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x);
		//}
		//return C;
	}

	Matrix GPUMultiplyA(Matrix B) {  //multiplies two matrices on the GPU where the first of which is effectively transposed before carrAyying out the calculation

		Matrix C(x, B.x);

		if (y != B.y) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}


		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (x * B.x <= MatrixGlobals::MaxThreads) {
			ThreadMode = true;
		}

		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = x;
		Thread.z = 1;//1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread
		//can only parallel

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.x;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadModeA << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, y);
		}
		else {
			Thread.y = 1;
			Block.x = B.x;
			Kernel::matrixMultiplicationKernelA << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, y);
		}
		return C;
	}

	Matrix GPUMultiplyB(Matrix B) { //multiplies two matrices on the GPU where the 2nd of which is effectively transposed before carrAyying out the calculation

		Matrix C(y, B.y);

		if (x != B.x) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}

		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (y * B.y <= MatrixGlobals::MaxThreads) {
			ThreadMode = true;
		}
		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = y;
		Thread.z = 1; //1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread
		 //can only parallel

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.y;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadModeB << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, B.y);
		}
		else {
			Thread.y = 1;
			Block.x = B.y;
			Kernel::matrixMultiplicationKernelB << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, B.y);
		}
		return C;
	}

	Matrix GPUMultiplyAB(Matrix B) { //multiplies two matrices on the GPU where both of which is effectively transposed before carrAyying out the calculation

		Matrix C(x, B.y);

		if (y != B.x) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}

		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (x * B.y <= MatrixGlobals::MaxThreads) {
			ThreadMode = true;
		}
		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = x;
		Thread.z = 1; //1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread
		//can only parallel

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.y;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadModeAB << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, y, B.y);
		}
		else {
			Thread.y = 1;
			Block.x = B.y;
			Kernel::matrixMultiplicationKernelAB << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, y, B.y);
		}
		return C;
	}

	void GPUMultiplyNoRet(Matrix B, Matrix C) { //multiplies two matrices on the GPU 
		if (x != B.y) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}


		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (y * B.x <= MatrixGlobals::MaxThreads) {
			ThreadMode = true;
		}

		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = y;
		Thread.z = 1; //1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.x;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadMode << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x);
		}
		else {
			Thread.y = 1;
			Block.x = B.x;
			Kernel::matrixMultiplicationKernel << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x);
		}

	}


	void GPUMultiplyANoRet(Matrix B, Matrix C) {  //multiplies two matrices on the GPU where the first of which is effectively transposed before carrAyying out the calculation
		if (y != B.y) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}


		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (x * B.x <= MatrixGlobals::MaxThreads) {
			ThreadMode = true;
		}

		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = x;
		Thread.z = 1;//1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread
		//can only parallel

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.x;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadModeA << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, y);
		}
		else {
			Thread.y = 1;
			Block.x = B.x;
			Kernel::matrixMultiplicationKernelA << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, y);
		}

	}

	void GPUMultiplyBNoRet(Matrix B, Matrix C) { //multiplies two matrices on the GPU where the 2nd of which is effectively transposed before carrAyying out the calculation
		if (x != B.x) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}

		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (y * B.y <= MatrixGlobals::MaxThreads) {
			ThreadMode = true;
		}
		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = y;
		Thread.z = 1; //1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread
		 //can only parallel

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.y;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadModeB << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, B.y);
		}
		else {
			Thread.y = 1;
			Block.x = B.y;
			Kernel::matrixMultiplicationKernelB << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, B.y);
		}
	}

	void GPUMultiplyABNoRet(Matrix B, Matrix C) { //multiplies two matrices on the GPU where both of which is effectively transposed before carrAyying out the calculation
		if (y != B.x) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}

		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (x * B.y <= MatrixGlobals::MaxThreads) {
			ThreadMode = true;
		}
		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = x;
		Thread.z = 1; //1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread
		//can only parallel

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.y;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadModeAB << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, y, B.y);
		}
		else {
			Thread.y = 1;
			Block.x = B.y;
			Kernel::matrixMultiplicationKernelAB << < Block, Thread >> > (arrAy, B.arrAy, C.arrAy, x, B.x, C.x, y, B.y);
		}

	}


	//====================================================================================================================================================================================






















	//---------------------------------------------------------------------    CPU Functions    ------------------------------------------------------------------------------------------

	double CPUIndex(size_t Y, size_t X) { //returns the value stored at that index
		return arrAy[Y * x + X];
	}

	double CPUIndexVect(size_t X) {
		return arrAy[X];
	}

	void CPUSetIndex(size_t Y, size_t X, double Val) {//gets the contents of the index requested from the matrix arrAy
		arrAy[Y * x + X] = Val;
	}

	void CPUSetIndexVect(size_t X, double Val) {
		arrAy[X] = Val;
	}

	void CPUAppendIndex(size_t Y, size_t X, double Val) {//gets the contents of the index requested from the matrix arrAy
		arrAy[Y * x + X] += Val;
	}

	void CPUAppendIndexVect(size_t X, double Val) {
		arrAy[X] += Val;
	}

	void CPURandomFill(double min, double max) { //fills the matrix with random real values in the range min to max
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> RandDist(min, max);
		for (int index = 0; index < x * y; index++) {
			arrAy[index] = RandDist(mt);
		}
	}

	void CPURandomFillDouble(double negmin, double negmax, double min, double max) { //fills the matrix with random real values between the two ranges
			//one negative, one positive, allowing the exclusion of 0 from the potential values stored in the arrAy
			//this is used for the weights to prevent any from getting "stuck" at 0 before the network has begun to converge
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> SignRandDist(0, 2);
		std::uniform_real_distribution<double> RandDist(min, max);
		for (int index = 0; index < x * y; index++) {
			if (SignRandDist(mt) > 1) {

				arrAy[index] = RandDist(mt);
			}
			else {
				arrAy[index] = 0 - RandDist(mt);
			}
		}
	}

	void CPUTranspose() { //transposes the matrix, by switching the positions of every element in the arrAy (except 0,0)
				//such that the rows before are stored as columns after
		Matrix C(x, y);
		for (int row = 0; row < x; row++) {
			for (int col = 0; col < y; col++) {
				C.SetIndex(row, col, Index(col, row));
			}
		}
		memcpy(arrAy, C.arrAy, size * sizeof(double));
		x = C.x;
		y = C.y;
	}

	Matrix CPUTransposeReturning() { //same as above but returns the transpose rather than applying the method to itself
		Matrix C(x, y);
		for (int row = 0; row < x; row++) {
			for (int col = 0; col < y; col++) {
				C.SetIndex(row, col, Index(col, row));
			}
		}
		return C;
	}

	Matrix CPUMultiply(Matrix B) { //multiply two matrices on the CPU
		if (x != B.y) {
			throw "Cannot multiply these";
		}
		Matrix C(y, B.x);
		if ((C.arrAy[0] != 0) | (C.arrAy[C.size - 1] != 0)) { //perform a rudimentary check of the initialization value
				//if the first and last terms are 0 they all will be so we can avoid the large number of iterations filling in the C with 0's
		}
		for (size_t row1 = 0; row1 < y; row1++) { //Multiply the matrices in a standard nested for loop executing in series
			for (size_t col2 = 0; col2 < B.x; col2++) {
				for (size_t col1 = 0; col1 < x; col1++) {
					C.AppendIndex(row1, col2, Index(row1, col1) * B.Index(col1, col2)); //appending the products to the same index in each of the lowest iterations
				}
			}
		}
		return C;
	}

	Matrix CPUMultiplyA(Matrix B) { //multiply two matrices on the CPU where the first given is transposed
		if (y != B.y) {
			throw "Cannot multiply these";
		}
		Matrix C(x, B.x);
		for (size_t row1 = 0; row1 < x; row1++) {
			for (size_t col2 = 0; col2 < B.x; col2++) {
				for (size_t col1 = 0; col1 < y; col1++) {
					C.AppendIndex(row1, col2, Index(col1, row1) * B.Index(col1, col2));
					//*C.Index(row1, col2) += *A.Index(col1, row1) * *B.Index(col1, col2);
				}
			}
		}
		return C;
	}

	Matrix CPUMultiplyB(Matrix B) { //multiply two matrices on the CPU where the second given is transposed
		if (x != B.x) {
			throw "Cannot multiply these";
		}
		Matrix C(y, B.y);
		for (size_t row1 = 0; row1 < y; row1++) {
			for (size_t col2 = 0; col2 < B.y; col2++) {
				for (size_t col1 = 0; col1 < x; col1++) {
					C.AppendIndex(row1, col2, Index(row1, col1) * B.Index(col2, col1));
					//*C.Index(row1, col2) += *Index(row1, col1) * *B.Index(col2, col1);
				}
			}
		}
		return C;
	}

	Matrix CPUMultiplyAB(Matrix B) { //multiply two matrices on the CPU where the second given is transposed
		if (y != B.y) {
			throw "Cannot multiply these";
		}
		Matrix C(x, B.x);
		for (size_t row1 = 0; row1 < x; row1++) {
			for (size_t col2 = 0; col2 < B.x; col2++) {
				for (size_t col1 = 0; col1 < y; col1++) {
					C.AppendIndex(row1, col2, Index(col1, row1) * B.Index(col1, col2));
				}
			}
		}
		return C;
	}

	void CPUMultiplyNoRet(Matrix B, Matrix C) { //multiply two matrices on the CPU
		if (x != B.y) {
			throw "Cannot multiply these";
		}
		//if ((C.arrAy[0] != 0) | (C.arrAy[C.size - 1] != 0)) { //perform a rudimentary check of the initialization value
				//if the first and last terms are 0 they all will be so we can avoid the large number of iterations filling in the C with 0's
		//}
		for (size_t row1 = 0; row1 < y; row1++) { //Multiply the matrices in a standard nested for loop executing in series
			for (size_t col2 = 0; col2 < B.x; col2++) {
				for (size_t col1 = 0; col1 < x; col1++) {
					C.AppendIndex(row1, col2, Index(row1, col1) * B.Index(col1, col2)); //appending the products to the same index in each of the lowest iterations
				}
			}
		}
	}

	void CPUMultiplyANoRet(Matrix B, Matrix C) { //multiply two matrices on the CPU where the first given is transposed
		if (y != B.y) {
			throw "Cannot multiply these";
		}
		for (size_t row1 = 0; row1 < x; row1++) {
			for (size_t col2 = 0; col2 < B.x; col2++) {
				for (size_t col1 = 0; col1 < y; col1++) {
					C.AppendIndex(row1, col2, Index(col1, row1) * B.Index(col1, col2));
					//*C.Index(row1, col2) += *A.Index(col1, row1) * *B.Index(col1, col2);
				}
			}
		}
	}

	void CPUMultiplyBNoRet(Matrix B, Matrix C) { //multiply two matrices on the CPU where the second given is transposed
		if (x != B.x) {
			throw "Cannot multiply these";
		}
		for (size_t row1 = 0; row1 < y; row1++) {
			for (size_t col2 = 0; col2 < B.y; col2++) {
				for (size_t col1 = 0; col1 < x; col1++) {
					C.AppendIndex(row1, col2, Index(row1, col1) * B.Index(col2, col1));
					//*C.Index(row1, col2) += *Index(row1, col1) * *B.Index(col2, col1);
				}
			}
		}
	}

	void CPUMultiplyABNoRet(Matrix B, Matrix C) { //multiply two matrices on the CPU where the second given is transposed
		if (y != B.y) {
			throw "Cannot multiply these";
		}
		for (size_t row1 = 0; row1 < x; row1++) {
			for (size_t col2 = 0; col2 < B.x; col2++) {
				for (size_t col1 = 0; col1 < y; col1++) {
					C.AppendIndex(row1, col2, Index(col1, row1) * B.Index(col1, col2));
				}
			}
		}
	}

	Matrix CPUAdd(Matrix B) { //adds two matrices together, computing the element-wise sum
		Matrix C(y, x);
		for (int index = 0; index < size; index++) {
			C.arrAy[index] = arrAy[index] + B.arrAy[index];
		}
		return C;
	}

	void CPUAddNoRet(Matrix B) { //adds two matrices together, computing the element-wise sum
		for (int index = 0; index < size; index++) {
			arrAy[index] += B.arrAy[index];
		}
	}

	void CPUAddNoRetCGiven(Matrix B, Matrix C) { //adds two matrices together, computing the element-wise sum
		for (int index = 0; index < size; index++) {
			C.arrAy[index] = arrAy[index] + B.arrAy[index];
		}
	}

	Matrix CPUAddConst(double B) { //adds a constant to every value in the given matrix
		Matrix C(y, x);
		for (int index = 0; index < size; index++) {
			C.arrAy[index] = arrAy[index] + B;
		}
		return C;
	}

	void CPUAddConstNoRet(double B) { //adds a constant to every value in the given matrix
		for (int index = 0; index < size; index++) {
			arrAy[index] += B;
		}
	}

	void CPUAddConstNoRetCGiven(double B, Matrix C) {
		for (int index = 0; index < size; index++) {
			C.arrAy[index] = arrAy[index] + B;
		}
	}

	Matrix CPUSigmoid() { //maps the matrix through the sigmoid function
		Matrix C(y, x);
		for (int index = 0; index < C.size; index++) {
			//auto s = std::exp(-arrAy[index]);
			C.arrAy[index] = 1 / (1 + std::exp(-arrAy[index]));
		}
		return C;
	}

	void CPUSigmoidNoRet() { //maps the matrix through the sigmoid function

		for (int index = 0; index < size; index++) {
			//auto s = std::exp(-arrAy[index]);
			arrAy[index] = 1 / (1 + std::exp(-arrAy[index]));
		}

	}

	void CPUSigmoidNoRetCGiven(Matrix C) { //maps the matrix through the sigmoid function

		for (int index = 0; index < size; index++) {
			//auto s = std::exp(-arrAy[index]);
			C.arrAy[index] = 1 / (1 + std::exp(-arrAy[index]));
		}

	}

	Matrix CPUScale(double ScalingFactor) { //scales the matrix by a given constant
		Matrix C(y, x);
		for (int i = 0; i < size; i++) {
			C.arrAy[i] = arrAy[i] * ScalingFactor;
		}
		return C;
	}

	void CPUScaleNoRet(double ScalingFactor) { //scales the matrix by a given constant
		for (int i = 0; i < size; i++) {
			arrAy[i] *= ScalingFactor;
		}
	}

	void CPUScaleNoRetCGiven(double ScalingFactor, Matrix C) {
		for (int i = 0; i < size; i++) {
			C.arrAy[i] = arrAy[i] * ScalingFactor;
		}
	}

	Matrix ElementWiseCPU(Matrix B) { //multiplies two matrices element by element, as if vectors
		Matrix C(y, x);
		for (int i = 0; i < size; i++) {
			C.arrAy[i] = arrAy[i] * B.arrAy[i];
		}
		return C;
	}

	void ElementWiseCPUNoRet(Matrix B) { //multiplies two matrices element by element, as if vectors
		for (int i = 0; i < size; i++) {
			arrAy[i] *= B.arrAy[i];
		}
	}

	void ElementWiseCPUNoRetCGiven(Matrix B, Matrix C) { //multiplies two matrices element by element, as if vectors
		for (int i = 0; i < size; i++) {
			C.arrAy[i] = arrAy[i] * B.arrAy[i];
		}
	}

	Matrix CPUCopy() {
		Matrix C(x, y);
		memcpy(C.arrAy, arrAy, size * sizeof(double));
		return C;
	}



	//===================================================================================================================================================================================














	//--------------------------------------------------------------------------    std::functions    -----------------------------------------------------------------------------------
	std::function<double(Matrix*, size_t Y, size_t X)> StdIndex = &Matrix::CPUIndex;
	std::function<double(Matrix*, size_t X)> StdIndexVect = &Matrix::CPUIndexVect;

	std::function<void(Matrix*, size_t X, size_t Y, double Val)> StdSetIndex = &Matrix::CPUSetIndex;
	std::function<void(Matrix*, size_t X, double Val)> StdSetIndexVect = &Matrix::CPUSetIndexVect;
	std::function<void(Matrix*)> StdTranspose = &Matrix::CPUTranspose;
	std::function<void(Matrix*, double Min, double Max)> StdRandomFill = &Matrix::CPURandomFill;
	std::function<void(Matrix*, double negmin, double negmax, double min, double max)> StdRandomFillDouble = &Matrix::CPURandomFillDouble;
	std::function<Matrix(Matrix*)> StdTransposeReturning = &Matrix::CPUTransposeReturning;
	std::function<void(Matrix*, size_t Y, size_t X, double Val)> StdAppendIndex = &Matrix::CPUAppendIndex;
	std::function<void(Matrix*, size_t X, double Val)> StdAppendIndexVect = &Matrix::CPUAppendIndexVect;

	std::function<Matrix(Matrix*, Matrix B)> StdMultiply = &Matrix::CPUMultiply;
	std::function<Matrix(Matrix*, Matrix B)> StdMultiplyA = &Matrix::CPUMultiplyA;
	std::function<Matrix(Matrix*, Matrix B)> StdMultiplyB = &Matrix::CPUMultiplyB;
	std::function<Matrix(Matrix*, Matrix B)> StdMultiplyAB = &Matrix::CPUMultiplyAB;

	std::function<void(Matrix*, Matrix B, Matrix C)> StdMultiplyNoRet = &Matrix::CPUMultiplyNoRet;
	std::function<void(Matrix*, Matrix B, Matrix C)> StdMultiplyANoRet = &Matrix::CPUMultiplyANoRet;
	std::function<void(Matrix*, Matrix B, Matrix C)> StdMultiplyBNoRet = &Matrix::CPUMultiplyBNoRet;
	std::function<void(Matrix*, Matrix B, Matrix C)> StdMultiplyABNoRet = &Matrix::CPUMultiplyABNoRet;

	std::function<Matrix(Matrix*, Matrix B)> StdAdd = &Matrix::CPUAdd;
	std::function<void(Matrix*, Matrix B)> StdAddNoRet = &Matrix::CPUAddNoRet;
	std::function<void(Matrix*, Matrix B, Matrix C)> StdAddNoRetCGiven = &Matrix::CPUAddNoRetCGiven;

	std::function<Matrix(Matrix*, double B)> StdAddConst = &Matrix::CPUAddConst;
	std::function<void(Matrix*, double B)> StdAddConstNoRet = &Matrix::CPUAddConstNoRet;
	std::function<void(Matrix*, double B, Matrix C)> StdAddConstNoRetCGiven = &Matrix::CPUAddConstNoRetCGiven;

	std::function<Matrix(Matrix*)> StdSigmoid = &Matrix::CPUSigmoid;
	std::function<void(Matrix*)> StdSigmoidNoRet = &Matrix::CPUSigmoidNoRet;
	std::function<void(Matrix*, Matrix C)> StdSigmoidNoRetCGiven = &Matrix::CPUSigmoidNoRetCGiven;

	std::function<Matrix(Matrix*, double B)> StdScale = &Matrix::CPUScale;
	std::function<void(Matrix*, double B)> StdScaleNoRet = &Matrix::CPUScaleNoRet;
	std::function<void(Matrix*, double B, Matrix C)> StdScaleNoRetCGiven = &Matrix::CPUScaleNoRetCGiven;

	std::function<Matrix(Matrix*, Matrix B)> StdElementWise = &Matrix::ElementWiseCPU;
	std::function<void(Matrix*, Matrix B)> StdElementWiseNoRet = &Matrix::ElementWiseCPUNoRet;
	std::function<void(Matrix*, Matrix B, Matrix C)> StdElementWiseNoRetCGiven = &Matrix::ElementWiseCPUNoRetCGiven;
	std::function<Matrix(Matrix*)> StdCopy = &Matrix::CPUCopy;
	//=========================================================================================================================================================================\==========



















	void SwitchMethods() {
		StdIndex = &Matrix::GPUIndex;
		StdIndexVect = &Matrix::GPUIndexVect;
		StdSetIndex = &Matrix::GPUSetIndex;
		StdSetIndexVect = &Matrix::GPUSetIndexVect;
		StdRandomFill = &Matrix::GPURandomFill;
		StdRandomFillDouble = &Matrix::GPURandomFillDouble;
		StdTranspose = &Matrix::GPUTranspose;
		StdTransposeReturning = &Matrix::GPUTransposeReturning;
		StdAppendIndex = &Matrix::GPUAppendIndex;
		StdAppendIndexVect = &Matrix::GPUAppendVect;

		StdMultiply = &Matrix::GPUMultiply;
		StdMultiplyA = &Matrix::GPUMultiplyA;
		StdMultiplyB = &Matrix::GPUMultiplyB;
		StdMultiplyAB = &Matrix::GPUMultiplyAB;

		StdMultiplyNoRet = &Matrix::GPUMultiplyNoRet;
		StdMultiplyANoRet = &Matrix::GPUMultiplyANoRet;
		StdMultiplyBNoRet = &Matrix::GPUMultiplyBNoRet;
		StdMultiplyABNoRet = &Matrix::GPUMultiplyABNoRet;
	}


public:
	double* arrAy; //Pointer to an arrAy on the device
	int y; //Number of rows in the matrix
	int x; //Number of columns in the matrix
	size_t size; //The size of the matrix (x*y)



	Matrix() { //Null constructor
		arrAy = nullptr;
		if (!MatrixGlobals::HasCheckedForGPU) {
			HasGPU();
		}
		if (MatrixGlobals::GPUPresent) {
			SwitchMethods();
		}
	}
	Matrix(size_t Y, size_t X) {//standard constructor taking the number of columns in the matrix and the number of rows
			//then allocating the appropriate amount of memory on the device
		if (!MatrixGlobals::HasCheckedForGPU) {
			HasGPU();
		}
		if (MatrixGlobals::GPUPresent) {
			size = X * Y;
			cudaMalloc(&arrAy, size * sizeof(double));
			y = Y;
			x = X;
		}
		else {
			size = X * Y;
			arrAy = (double*)malloc(size * sizeof(double));
			y = Y;
			x = X;
		}
	}
	~Matrix() {
		if (MatrixGlobals::GPUPresent) {
#ifdef GPU
			cudaFree(arrAy);
#endif
		}
		else {
			//free(arrAy);
		}
	}

	inline size_t GetIndex(size_t Y, size_t X) { //gets the index of a given co-ordinate in the arrAy
		return Y * x + X;
	}

	inline double Index(size_t Y, size_t X) {
		return StdIndex(this, Y, X);
	}
	inline double Index(size_t X) {
		return StdIndexVect(this, X);
	}
	inline void SetIndex(size_t Y, size_t X, double Val) {
		StdSetIndex(this, Y, X, Val);
	}
	inline void SetIndex(size_t X, double Val) {
		StdSetIndexVect(this, X, Val);
	}
	void Fill(double Val) { //fills the matrix with a given value
		for (int index = 0; index < x * y; index++) {
			SetIndex(index, Val);
		}
	}
	inline void AppendIndex(size_t Y, size_t X, double Val) {
		StdAppendIndex(this, Y, X, Val);
	}
	inline void AppendIndex(size_t X, double Val) {
		StdAppendIndexVect(this, X, Val);
	}
	//-----------------------------------------------------------------    Print Methods    -------------------------------------------------------------------------------------------------

	void print() { //prints the matrix to the console
		for (int c = 0; c < y; c++) {
			for (int r = 0; r < x; r++) {
				std::cout << Index(c, r) << " ";
			}
			std::cout << std::endl;
		}
	}

	void printHighRes() { //prints the matrix to the console with 25 digits of precision
		std::streamsize ss = std::cout.precision();
		std::cout.precision(25);
		for (int c = 0; c < y; c++) {
			for (int r = 0; r < x; r++) {
				std::cout << std::fixed << Index(c, r) << " ";
			}
			std::cout << std::endl;
		}
		std::cout.precision(ss);
	}

	void printHighRes(int Resolution) { //print the matrix to the console with a defined number of digits
		std::streamsize ss = std::cout.precision();
		std::cout.precision(Resolution);
		for (int c = 0; c < y; c++) {
			for (int r = 0; r < x; r++) {
				std::cout << std::fixed << Index(c, r) << "			";
			}
			std::cout << std::endl;
		}
		std::cout.precision(ss);
	}
	//=======================================================================================================================================================================================

	inline Matrix Multiply(Matrix B) {
		return StdMultiply(this, B);
	}
	inline Matrix MultiplyA(Matrix B) {
		return StdMultiplyA(this, B);
	}
	inline Matrix MultiplyB(Matrix B) {
		return StdMultiplyB(this, B);
	}
	inline Matrix MultiplyAB(Matrix B) {
		return StdMultiplyAB(this, B);
	}
	inline void Multiply(Matrix B, Matrix C) {
		StdMultiplyNoRet(this, B, C);
	}
	inline void MultiplyA(Matrix B, Matrix C) {
		StdMultiplyANoRet(this, B, C);
	}
	inline void MultiplyB(Matrix B, Matrix C) {
		StdMultiplyBNoRet(this, B, C);
	}
	inline void MultiplyAB(Matrix B, Matrix C) {
		StdMultiplyABNoRet(this, B, C);
	}



	inline void RandomFill(double min, double max) {
		StdRandomFill(this, min, max);
	}
	inline void RandomFill(double negmin, double negmax, double min, double max) {
		StdRandomFillDouble(this, negmin, negmax, min, max);
	}

	inline Matrix Add(Matrix B) {
		return StdAdd(this, B);
	}

	inline void AddNoRet(Matrix B) {
		StdAddNoRet(this, B);
	}

	inline void AddNoRet(Matrix B, Matrix C) {
		StdAddNoRetCGiven(this, B, C);
	}

	inline Matrix AddConst(double B) {
		return StdAddConst(this, B);
	}

	inline void AddConstNoRet(double B) {
		StdAddConstNoRet(this, B);
	}

	inline void AddConstNoRet(double B, Matrix C) {
		StdAddConstNoRetCGiven(this, B, C);
	}

	inline Matrix Sigmoid() {
		return StdSigmoid(this);
	}

	inline void SigmoidNoRet() {
		StdSigmoidNoRet(this);
	}

	inline void SigmoidNoRet(Matrix C) {
		StdSigmoidNoRetCGiven(this, C);
	}

	inline Matrix Scale(double ScalingFactor) {
		StdScale(this, ScalingFactor);
	}

	inline void ScaleNoRet(double ScalingFactor) {
		StdScaleNoRet(this, ScalingFactor);
	}

	inline void ScaleNoRet(double ScalingFactor, Matrix C) {
		StdScaleNoRetCGiven(this, ScalingFactor, C);
	}

	inline Matrix ElementWise(Matrix B) {
		return StdElementWise(this, B);
	}
	inline void ElementWiseNoRet(Matrix B) {
		StdElementWiseNoRet(this, B);
	}

	inline void ElementWiseNoRet(Matrix B, Matrix C) {
		StdElementWiseNoRetCGiven(this, B, C);
	}

};

#ifdef GPU
class MatrixGPU { //Matrix class stored as an arrAy of doubles on the GPU
public:
	double* arrAy; //Pointer to an arrAy on the device
	int y; //Number of rows in the matrix
	int x; //Number of columns in the matrix
	size_t size; //The size of the matrix (x*y)

	MatrixGPU() { //Null constructor used if the matrix is to be initialized manually
		arrAy = nullptr;
	}
	MatrixGPU(int Y, int X) {
		size = X * Y;
		cudaMalloc(&arrAy, size * sizeof(double));
		y = Y;
		x = X;
	}
	double Index(size_t Y, size_t X) {//Gets a value from the matrix indexed in both dimensions
		double ret = 0;
		cudaMemcpy(&ret, &arrAy[Y * x + X], sizeof(double), cudaMemcpyDeviceToHost); //copy's the value from GPU memory to host memory
		return ret;
	}
	double Index(size_t X) {//Gets a value from the matrix indexed by a single dimension 0 to Matrix size
		double ret = 0;
		cudaMemcpy(&ret, &arrAy[X], sizeof(double), cudaMemcpyDeviceToHost); //copy's the value from GPU memory to host memory
		return ret;
	}
	size_t GetIndex(size_t Y, size_t X) { //gets the index of a given co-ordinate in the arrAy
		return Y * x + X;
	}
	void SetIndex(size_t Y, size_t X, double Val) { //set the value of an co-ordinate in the GPU matrix by copying the parameter from the CPU to the GPU device at that memory location
		double temp = Val;
		cudaMemcpy(&arrAy[Y * x + X], &temp, sizeof(double), cudaMemcpyHostToDevice);
	}
	void SetIndex(size_t X, double Val) { //set the value of an index in the GPU matrix arrAy by copying the parameter from the CPU to the GPU device at that memory location
		double temp = Val;
		cudaMemcpy(&(arrAy[X]), &temp, sizeof(double), cudaMemcpyHostToDevice);
	}
	void CPUTranspose() { //transpose the matrix on the CPU - this takes a very long time for larger matrices due to the number of retrievals and stores involved, but was on;y used for testing
		MatrixGPU C(x, y);
		for (int row = 0; row < y; row++) {
			for (int col = 0; col < x; col++) {
				C.SetIndex(col, row, Index(row, col));
			}
		}
		this->arrAy = C.arrAy;
		this->x = C.x;
		this->y = C.y;
	}

	void RandomFill(double min, double max) { //fills the matrix with a random values in the desired range
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> RandDist(min, max);
		double* TemporaryarrAy;
		TemporaryarrAy = (double*)malloc(sizeof(double) * size);
		for (int index = 0; index < size; index++) {
			TemporaryarrAy[index] = RandDist(mt);
		}
		cudaMemcpy(arrAy, TemporaryarrAy, sizeof(double) * size, cudaMemcpyHostToDevice);
		free(TemporaryarrAy);

	}
	void RandomFill(double negmin, double negmax, double min, double max) { //fills the matrix with random values in the given ranges, used to exclude 0's from the range required for weights and biases
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> SignRandDist(0, 2);
		std::uniform_real_distribution<double> RandDist(min, max);
		double* TemporaryarrAy;
		TemporaryarrAy = (double*)malloc(sizeof(double) * size);
		for (int index = 0; index < size; index++) {
			if (SignRandDist(mt) > 1) {
				TemporaryarrAy[index] = RandDist(mt);
			}
			else {
				TemporaryarrAy[index] = 0 - RandDist(mt);
			}
		}
		cudaMemcpy(arrAy, TemporaryarrAy, sizeof(double) * size, cudaMemcpyHostToDevice);
		free(TemporaryarrAy);
	}

	void Fill(double Val) { //fills the matrix with a given value
		for (int index = 0; index < x * y; index++) {
			SetIndex(index, Val);
		}
	}

	void print() { //prints the matrix to the console
		for (int c = 0; c < y; c++) {
			for (int r = 0; r < x; r++) {
				std::cout << Index(c, r) << " ";
			}
			std::cout << std::endl;
		}
	}

	void printHighRes() { //prints the matrix to the console with 25 digits of precision
		std::streamsize ss = std::cout.precision();
		std::cout.precision(25);
		for (int c = 0; c < y; c++) {
			for (int r = 0; r < x; r++) {
				std::cout << std::fixed << Index(c, r) << " ";
			}
			std::cout << std::endl;
		}
		std::cout.precision(ss);
	}

	void printHighRes(int Resolution) { //print the matrix to the console with a defined number of digits
		std::streamsize ss = std::cout.precision();
		std::cout.precision(Resolution);
		for (int c = 0; c < y; c++) {
			for (int r = 0; r < x; r++) {
				std::cout << std::fixed << Index(c, r) << "			";
			}
			std::cout << std::endl;
		}
		std::cout.precision(ss);
	}
};

int CreateCublasHandle(cublasHandle_t* Handle) { //Create a handle to a CUBALAS accelerated compute API
	cublasStatus_t Status = CUBLAS_STATUS_EXECUTION_FAILED;
	Status = cublasCreate(Handle);
	while (Status != CUBLAS_STATUS_SUCCESS) {
		Status = cublasCreate(Handle);
		if (Status != CUBLAS_STATUS_SUCCESS) {
			printf("CUBLAS initialization failed\n");
			if (Handle != nullptr) {
				cublasDestroy(*Handle);
			}

		}
	}
	return EXIT_SUCCESS;
}
#endif // GPU



class MatrixCPU { //Matrix class stored as an arrAy of doubles on the CPU
public:
	std::vector<double> arrAy; //Pointer to an arrAy on the device
	size_t y; //Number of rows in the matrix
	size_t x; //Number of columns in the matrix
	size_t size; //The size of the matrix (x*y)
	MatrixCPU() { //Null constructor used if the matrix is to be initialized manually
		arrAy.resize(0);
	}
	MatrixCPU(int Y, int X) { //standard constructor taking the number of columns in the matrix and the number of rows
			//then constructing a new vector arrAy to contain the matrix data, thus allocating sufficient memory
		size = X * Y;
		arrAy.resize(size);
		y = Y;
		x = X;
	}

	auto Index(size_t Y, size_t X) { //returns a pointer to the value stored at that index
		return &arrAy[Y * x + X];
	}
	void CPUTranspose() { //transposes the matrix, by switching the positions of every element in the arrAy (except 0,0)
				//such that the rows before are stored as columns after
		MatrixCPU C(x, y);
		for (int row = 0; row < y; row++) {
			for (int col = 0; col < x; col++) {
				*C.Index(col, row) = *(this->Index(row, col));
			}
		}
		this->arrAy = C.arrAy;
		this->x = C.x;
		this->y = C.y;
	}
	MatrixCPU Transpose() { //same as above but returns the transpose rather than applying the method to itself
		MatrixCPU C(x, y);
		for (int row = 0; row < x; row++) {
			for (int col = 0; col < y; col++) {
				*C.Index(row, col) = *(this->Index(col, row));
			}
		}
		return C;
	}
	double Get(size_t Y, size_t X) {//gets the contents of the index requested from the matrix arrAy
		return arrAy[Y * x + X];
	}
	void RandomFill(double min, double max) { //fills the matrix with random real values in the range min to max
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> RandDist(min, max);
		for (int index = 0; index < x * y; index++) {
			arrAy[index] = RandDist(mt);
		}
	}
	void RandomFill(double negmin, double negmax, double min, double max) { //fills the matrix with random real values between the two ranges
			//one negative, one positive, allowing the exclusion of 0 from the potential values stored in the arrAy
			//this is used for the weights to prevent any from getting "stuck" at 0 before the network has begun to converge
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> SignRandDist(0, 2);
		std::uniform_real_distribution<double> RandDist(min, max);
		for (int index = 0; index < x * y; index++) {
			if (SignRandDist(mt) > 1) {

				arrAy[index] = RandDist(mt);
			}
			else {
				arrAy[index] = 0 - RandDist(mt);
			}
		}
	}

	void Fill(double Val) { //fills the whole matrix with a single value, defined in the parameter
		for (int index = 0; index < x * y; index++) {
			arrAy[index] = Val;
		}
	}

	void print() { //prints the matrix row by row to the console
		for (int c = 0; c < y; c++) {
			for (int r = 0; r < x; r++) {
				std::cout << *Index(c, r) << " ";
			}
			std::cout << std::endl;
		}
	}
	void printHighRes() { //prints the matrix row by row to the console with a precision of 25 characters
		std::streamsize ss = std::cout.precision();
		std::cout.precision(25);
		for (int c = 0; c < y; c++) {
			for (int r = 0; r < x; r++) {
				std::cout << std::fixed << *Index(c, r) << " ";
			}
			std::cout << std::endl;
		}
		std::cout.precision(ss);
	}
	void printHighRes(int Resolution) { //prints the matrix row by row to the console with a precision defined in the parameter
		std::streamsize ss = std::cout.precision();
		std::cout.precision(Resolution);
		for (int c = 0; c < y; c++) {
			for (int r = 0; r < x; r++) {
				std::cout << std::fixed << *Index(c, r) << "			";
			}
			std::cout << std::endl;
		}
		std::cout.precision(ss);
	}
};




namespace MatrixMath {
	//CPU matrix mathematical functions

	MatrixCPU CPUMultiply(MatrixCPU A, MatrixCPU B) { //multiply two matrices on the CPU
		if (A.x != B.y) {
			throw "Cannot multiply these";
		}
		MatrixCPU C(A.y, B.x);
		if ((C.arrAy[0] != 0) | (C.arrAy[C.size - 1] != 0)) { //perform a rudimentary check of the initialization value
				//if the first and last terms are 0 they all will be so we can avoid the large number of iterations filling in the C with 0's
			C.Fill(0);
		}
		for (size_t row1 = 0; row1 < A.y; row1++) { //Multiply the matrices in a standard nested for loop executing in series
			for (size_t col2 = 0; col2 < B.x; col2++) {
				for (size_t col1 = 0; col1 < A.x; col1++) {
					*C.Index(row1, col2) += *A.Index(row1, col1) * *B.Index(col1, col2); //appending the products to the same index in each of the lowest iterations
				}
			}
		}
		return C;
	}


	MatrixCPU CPUMultiplyA(MatrixCPU A, MatrixCPU B) { //multiply two matrices on the CPU where the first given is transposed
		if (A.y != B.y) {
			throw "Cannot multiply these";
		}
		MatrixCPU C(A.x, B.x);
		C.Fill(0);
		for (size_t row1 = 0; row1 < A.x; row1++) {
			for (size_t col2 = 0; col2 < B.x; col2++) {
				for (size_t col1 = 0; col1 < A.y; col1++) {

					*C.Index(row1, col2) += *A.Index(col1, row1) * *B.Index(col1, col2);
				}
			}
		}
		return C;
	}

	MatrixCPU CPUMultiplyB(MatrixCPU A, MatrixCPU B) { //multiply two matrices on the CPU where the second given is transposed
		if (A.x != B.x) {
			throw "Cannot multiply these";
		}
		MatrixCPU C(A.y, B.y);
		C.Fill(0);
		for (size_t row1 = 0; row1 < A.y; row1++) {
			for (size_t col2 = 0; col2 < B.y; col2++) {
				for (size_t col1 = 0; col1 < A.x; col1++) {

					*C.Index(row1, col2) += *A.Index(row1, col1) * *B.Index(col2, col1);
				}
			}
		}
		return C;
	}

	MatrixCPU CPUMultiply(MatrixCPU A, std::complex<double>* B, int BSize) { //multiply two matrices on the CPU where the second is a arrAy of complex numbers
								//consequently it is treated as a matrix with dimensions, BSize * 1. Used during Live detection to reduce the number of data conversions
		if (A.x != BSize) {
			throw "Cannot multiply these";
		}
		MatrixCPU C(A.y, 1);
		C.Fill(0);
		for (size_t row1 = 0; row1 < A.y; row1++) {
			for (size_t col2 = 0; col2 < 1; col2++) {
				for (size_t col1 = 0; col1 < A.x; col1++) {

					*C.Index(row1, col2) += *A.Index(row1, col1) * abs(B[col1]);
				}
			}
		}
		return C;
	}

	MatrixCPU CPUAdd(MatrixCPU A, MatrixCPU B) { //adds two matrices together, computing the element-wise sum
		MatrixCPU C(A.y, A.x);
		for (int index = 0; index < A.size; index++) {
			C.arrAy[index] = A.arrAy[index] + B.arrAy[index];
		}
		return C;
	}

	MatrixCPU CPUAddConst(MatrixCPU A, double B) { //adds a constant to every value in the given matrix
		MatrixCPU C(A.y, A.x);
		for (int index = 0; index < A.size; index++) {
			C.arrAy[index] = A.arrAy[index] + B;
		}
		return C;
	}



	MatrixCPU Sigmoid(MatrixCPU A) { //maps the matrix through the sigmoid function
		MatrixCPU C(A.y, A.x);
		for (int index = 0; index < C.size; index++) {
			//auto s = std::exp(-A.arrAy[index]);
			C.arrAy[index] = 1 / (1 + std::exp(-A.arrAy[index]));
		}
		return C;
	}


	MatrixCPU CPUScale(MatrixCPU A, double ScalingFactor) { //scales the matrix by a given constant
		MatrixCPU C(A.y, A.x);
		for (int i = 0; i < A.size; i++) {
			C.arrAy[i] = A.arrAy[i] * ScalingFactor;
		}
		return C;
	}

	MatrixCPU ElementWiseCPU(MatrixCPU A, MatrixCPU B) { //multiplies two matrices element by element, as if vectors
		MatrixCPU C(A.y, A.x);
		for (int i = 0; i < A.size; i++) {
			C.arrAy[i] = A.arrAy[i] * B.arrAy[i];
		}
		return C;
	}
}


#ifdef GPU
namespace MatrixMath {
	//GPU matrix mathematical functions

	int FindMaxThreads() { //finds the number of available CUDA threads on the GPU device
		int nDevices;
		int ThreadCount = 0;
		cudaGetDeviceCount(&nDevices);
		for (int i = 0; i < nDevices; i++) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			int* threads = prop.maxThreadsDim;
			if (threads[0] > ThreadCount) {
				ThreadCount = threads[0];
			}
		}
		return ThreadCount;
	}

	void Fill(MatrixGPU A) { //fills a matrix on the GPU with a value
		for (int i = 0; i < A.size; i++) {
			A.SetIndex(i, 0);
		}
	}

	MatrixGPU CPUMultiply(MatrixGPU A, MatrixGPU B, MatrixGPU C) { //multiplies GPU matrices on the CPU by retrieving them from the device,
				//performing the calculation on the CPU and storing the result on the GPU in C
		if (A.x != B.y) {
			throw "Cannot multiply these";
		}
		C.Fill(0);
		for (size_t row1 = 0; row1 < A.y; row1++) {
			for (size_t col2 = 0; col2 < B.x; col2++) {
				for (size_t col1 = 0; col1 < A.x; col1++) {

					C.SetIndex(row1, col2, A.Index(row1, col1) * B.Index(col1, col2));

				}
			}
		}
		return C;
	}
	MatrixGPU CPUMultiplyA(MatrixGPU A, MatrixGPU B, MatrixGPU C) { //Same as CPUMultiply, but the A is transposed, by flipping the x/y indexing
		if (A.y != B.y) {
			throw "Cannot multiply these";
		}

		for (size_t row1 = 0; row1 < A.x; row1++) {
			for (size_t col2 = 0; col2 < B.x; col2++) {
				for (size_t col1 = 0; col1 < A.y; col1++) {

					C.SetIndex(row1, col2, A.Index(col1, row1) * B.Index(col1, col2));
				}
			}
		}
		return C;
	}

	MatrixGPU CPUAdd(MatrixGPU A, MatrixGPU B, MatrixGPU C) { //Adds two matrices on the GPU on the CPU retrieving each index and storing the result on the GPU in C
		for (int index = 0; index < A.size; index++) {
			C.arrAy[index] = A.arrAy[index] + B.arrAy[index];
		}
		return C;
	}

	MatrixGPU Sigmoid(MatrixGPU A, MatrixGPU C) { //Performs the Sigmoid function on a MatrixGPU on the GPU using the CPU, storing the result on the GPU in C

		for (int index = 0; index < C.size; index++) {
			//auto s = std::exp(-A.arrAy[index]);
			C.arrAy[index] = 1 / (1 + std::exp(-A.arrAy[index]));
		}
		return C;
	}

	void GPUtoCPU(MatrixGPU A, MatrixCPU* B) {
		B->size = A.size;
		B->arrAy.resize(A.size);
		for (int i = 0; i < A.size; i++) {
			B->arrAy[i] = A.Index(i);
		}
		B->y = A.y;
		B->x = A.x;
	}
	void CPUTranspose(MatrixGPU A, MatrixGPU C) { //transposes a GPU matrix using the CPU
		for (int row = 0; row < A.y; row++) {
			for (int col = 0; col < A.x; col++) {

				C.SetIndex(col, row, A.Index(row, col));
			}
		}
	}



	void Add(MatrixGPU A, MatrixGPU B, MatrixGPU C, cublasHandle_t handle) { //adds two matrices using the CUBLAS compute API
		double alp = 1;
		cudaError CopyError = cudaErrorLaunchFailure;
		while (CopyError != cudaSuccess) {
			CopyError = cudaMemcpy(C.arrAy, B.arrAy, A.size * sizeof(double), cudaMemcpyDeviceToDevice);
			if (CopyError != cudaSuccess) {
				std::cout << "Something went wrong with cuda while copying" << std::endl;
			}
		}
		cublasStatus_t Status = CUBLAS_STATUS_EXECUTION_FAILED;
		while (Status != CUBLAS_STATUS_SUCCESS) {
			Status = cublasDaxpy(handle, A.size, &alp, A.arrAy, 1, C.arrAy, 1);
			if (Status != CUBLAS_STATUS_SUCCESS) {
				std::cout << "Something went wrong with Cublas while adding" << std::endl;
				CreateCublasHandle(&handle);
			}
		}
	}


	void CopyMatrix(MatrixGPU Src, MatrixGPU Dest) { //copy's a matrix on the GPU
		Dest.x = Src.x;
		Dest.y = Src.y;
		Dest.size = Src.size;
		cudaMemcpy(Dest.arrAy, Src.arrAy, Src.size * sizeof(double), cudaMemcpyDeviceToDevice);
	}



	void AddConst(MatrixGPU A, double B, MatrixGPU C, int MaxThreads) { //Adds a constant to every element of the matrix given


		bool ThreadMode = false; //Maximizes the number of open threads => mini-mises compute time
		if (A.y * A.x <= MaxThreads) {
			ThreadMode = true;
		}
		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = A.y;
		Thread.z = 1;  //1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = A.x;
			Block.x = 1;
			Kernel::matrixAddThreadMode << < Block, Thread >> > (A.arrAy, B, C.arrAy, A.x);
		}
		else {
			Thread.y = 1;
			Block.x = A.x;
			Kernel::matrixAdd << < Block, Thread >> > (A.arrAy, B, C.arrAy, A.x);
		}

	}

	void Scale(MatrixGPU A, double B, MatrixGPU C, cublasHandle_t handle) { //scales a matrix by a constant value
		cudaError CopyError = cudaErrorLaunchFailure;
		while (CopyError != cudaSuccess) {
			CopyError = cudaMemcpy(C.arrAy, A.arrAy, A.size * sizeof(double), cudaMemcpyDeviceToDevice);
			if (CopyError != cudaSuccess) {
				std::cout << "Something went wrong with cuda while copying" << std::endl;
			}
		}
		cublasStatus_t Status = CUBLAS_STATUS_EXECUTION_FAILED;
		while (Status != CUBLAS_STATUS_SUCCESS) {
			Status = cublasDscal(handle, A.size, &B, C.arrAy, 1);
			if (Status != CUBLAS_STATUS_SUCCESS) {
				std::cout << "Something went wrong with Cublas while scaling" << std::endl;
				CreateCublasHandle(&handle);
			}
		}
	}

	void ElementWiseMultiply(MatrixGPU A, MatrixGPU B, MatrixGPU C, int MaxThreads) { //multiplies each element in one matrix by that in another

		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (A.y * A.x <= MaxThreads) {
			ThreadMode = true;
		}
		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = A.y;
		Thread.z = 1; //1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = A.x;
			Block.x = 1;
			Kernel::matrixElementWiseMultiplyThreadMode << < Block, Thread >> > (A.arrAy, B.arrAy, C.arrAy, A.x);
		}
		else {
			Thread.y = 1;
			Block.x = A.x;
			Kernel::matrixElementWiseMultiply << < Block, Thread >> > (A.arrAy, B.arrAy, C.arrAy, A.x);
		}

	}

	void SigmoidGPU(MatrixGPU A, MatrixGPU C) { //applies the sigmoid function to every element in a matrix
		dim3 dimGrid(512);
		size_t Size = A.size;
		int threadGrid = (Size + (dimGrid.x - 1)) / dimGrid.x;
		if (threadGrid > 65520) threadGrid = 65520;
		dim3 dimBlock(threadGrid);
		Kernel::GPUSigmoidarrAyKernel << <dimBlock, dimGrid >> > (A.arrAy, C.arrAy, Size);

	}


	void GPUMultiply(MatrixGPU A, MatrixGPU B, MatrixGPU C, int MaxThreads) { //multiplies two matrices on the GPU 
		if (A.x != B.y) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}


		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (A.y * B.x <= MaxThreads) {
			ThreadMode = true;
		}

		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = A.y;
		Thread.z = 1; //1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.x;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadMode << < Block, Thread >> > (A.arrAy, B.arrAy, C.arrAy, A.x, B.x, C.x);
		}
		else {
			Thread.y = 1;
			Block.x = B.x;
			Kernel::matrixMultiplicationKernel << < Block, Thread >> > (A.arrAy, B.arrAy, C.arrAy, A.x, B.x, C.x);
		}

	}


	void GPUMultiplyA(MatrixGPU A, MatrixGPU B, MatrixGPU C, int MaxThreads) {  //multiplies two matrices on the GPU where the first of which is effectively transposed before carrAyying out the calculation
		if (A.y != B.y) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}


		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (A.x * B.x <= MaxThreads) {
			ThreadMode = true;
		}

		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = A.x;
		Thread.z = 1;//1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread
		//can only parallel

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.x;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadModeA << < Block, Thread >> > (A.arrAy, B.arrAy, C.arrAy, A.x, B.x, C.x, A.y);
		}
		else {
			Thread.y = 1;
			Block.x = B.x;
			Kernel::matrixMultiplicationKernelA << < Block, Thread >> > (A.arrAy, B.arrAy, C.arrAy, A.x, B.x, C.x, A.y);
		}

	}

	void GPUMultiplyB(MatrixGPU A, MatrixGPU B, MatrixGPU C, int MaxThreads) { //multiplies two matrices on the GPU where the 2nd of which is effectively transposed before carrAyying out the calculation
		if (A.x != B.x) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}

		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (A.y * B.y <= MaxThreads) {
			ThreadMode = true;
		}
		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = A.y;
		Thread.z = 1; //1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread
		 //can only parallel

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.y;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadModeB << < Block, Thread >> > (A.arrAy, B.arrAy, C.arrAy, A.x, B.x, C.x, B.y);
		}
		else {
			Thread.y = 1;
			Block.x = B.y;
			Kernel::matrixMultiplicationKernelB << < Block, Thread >> > (A.arrAy, B.arrAy, C.arrAy, A.x, B.x, C.x, B.y);
		}
	}

	void GPUMultiplyAB(MatrixGPU A, MatrixGPU B, MatrixGPU C, int MaxThreads) { //multiplies two matrices on the GPU where both of which is effectively transposed before carrAyying out the calculation
		if (A.y != B.x) { //check that the matrices are correct size for multiplying
			throw "Cannot multiply these"; //throw exception if they are
		}

		bool ThreadMode = false; //Maximizes the number of open threads => minimizes compute time
		if (A.x * B.y <= MaxThreads) {
			ThreadMode = true;
		}
		dim3 Thread; //dimensions of the number of threads to start on the GPU
		Thread.x = A.x;
		Thread.z = 1; //1 indicating calling 1 x Grid.x
		dim3 Block; //Dimensions of the number of blocks to be called in each thread
		//can only parallel

		Block.y = 1;
		Block.z = 1;
		if (ThreadMode) {
			Thread.y = B.y;
			Block.x = 1;
			Kernel::matrixMultiplicationKernelThreadModeAB << < Block, Thread >> > (A.arrAy, B.arrAy, C.arrAy, A.x, B.x, C.x, A.y, B.y);
		}
		else {
			Thread.y = 1;
			Block.x = B.y;
			Kernel::matrixMultiplicationKernelAB << < Block, Thread >> > (A.arrAy, B.arrAy, C.arrAy, A.x, B.x, C.x, A.y, B.y);
		}

	}


}
#endif // GPU

