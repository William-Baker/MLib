#include "Matrix.hpp"
#include "kernel.cu"

cublasHandle_t* GPUMatrix::handle;
double GPUMatrix::SMThreads;
cudaStream_t GPUMatrix::stream; 

void GPUMatrix::GPUSupported(bool* supported) {
	GPUMatrix::handle = new cublasHandle_t();
	cublasStatus_t status = cublasCreate(GPUMatrix::handle);

	if (*GPUMatrix::handle == NULL && status != CUBLAS_STATUS_SUCCESS) {
		*supported = false;
	}
	else {
		*supported = true;
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		for (int i = 0; i < nDevices; i++) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			GPUMatrix::SMThreads = prop.maxThreadsPerBlock;
		}
		//cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
		int greatestPriority;
		cudaDeviceGetStreamPriorityRange(NULL, &greatestPriority);
		cudaStreamCreateWithPriority(&GPUMatrix::stream, cudaStreamDefault, greatestPriority);
		//cudaStreamCreateWithPriority(&GPUMatrix::stream, cudaStreamNonBlocking, 0);
		//cudaStreamCaptureMode cap = cudaStreamCaptureModeGlobal;
		cudaStreamCaptureMode cap = cudaStreamCaptureModeThreadLocal;

		cudaThreadExchangeStreamCaptureMode(&cap);
	}
}





GPUMatrix::GPUMatrix(const AbstractMatrix<double>* src) : GPUMatrix(){ 
	if(dynamic_cast<const CPUMatrix*>(src)){
		const CPUMatrix* actual = static_cast<const CPUMatrix*>(src);
		x = actual->x;
		y = actual->y;
		size = actual->get_size();
		arr = allocate_GPU_memory<double>(size);
		copy_GPU_memory(arr, actual->arr, size, cudaMemcpyHostToDevice);
	}
	else if(dynamic_cast<const GPUMatrix*>(src)){
		const GPUMatrix* actual = static_cast<const GPUMatrix*>(src);
		transfer(actual->copy());
	}
	else{
		ilog(FATAL_ERROR, "unknown source for copy constructor");
	}
}











//Functionality

GPUMatrix* GPUMatrix::multiply(AbstractMatrix* B) { //multiplies two matrices on the GPU 
	GPUMatrix* C = new GPUMatrix(y, B->x);
	multiply(B, C);
	return C;

}

void GPUMatrix::multiply(AbstractMatrix* B, AbstractMatrix* C) { //multiplies two matrices on the GPU 
	double proportion = sqrt(GPUMatrix::SMThreads / (y * B->x)); //calculate a proportion to devide up the threads,
								//since y*B->x may be less than threads guard used in kernel
	dim3 Thread(std::max(proportion * y, 1.0), std::max(proportion * B->x, 1.0), 1); //allocate proportions of threads
	dim3 Block(ceil((double)y / Thread.x), ceil((double)B->x / Thread.y), 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUMultKernel << < Block, Thread , 0, stream>> > (arr, B->arr, C->arr, C->y, C->x, x);
	//cudaStreamSynchronize(stream);
	
}

GPUMatrix* GPUMatrix::multiplyA(AbstractMatrix* B) {  //multiplies two matrices on the GPU where the first of which is effectively transposed before carrayying out the calculation
	GPUMatrix* C = new GPUMatrix(x, B->x);

	if (y != B->y) { //check that the matrices are correct size for multiplying
		throw "Cannot multiply these"; //throw exception if they are
	}
	multiplyA(B, C);
	return C;
}
void GPUMatrix::multiplyA(AbstractMatrix* B, AbstractMatrix* C) {
	double proportion = sqrt(GPUMatrix::SMThreads / (x * B->x)); //calculate a proportion to devide up the threads,
								//since y*B->x may be less than threads guard used in kernel
	dim3 Thread(std::max(proportion * x, 1.0), std::max(proportion * B->x, 1.0), 1); //allocate proportions of threads
	dim3 Block(ceil((double)x / Thread.x), ceil((double)B->x / Thread.y), 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUMultKernelA << < Block, Thread , 0, stream>> > (arr, B->arr, C->arr, C->y, C->x, y);
	//cudaStreamSynchronize(stream);
}

GPUMatrix* GPUMatrix::multiplyB(AbstractMatrix* B) {  //multiplies two matrices on the GPU where the first of which is effectively transposed before carrayying out the calculation
	GPUMatrix* C = new GPUMatrix(y, B->y);

	if (x != B->x) { //check that the matrices are correct size for multiplying
		throw "Cannot multiply these"; //throw exception if they are
	}
	multiplyB(B, C);
	return C;
}
void GPUMatrix::multiplyB(AbstractMatrix* B, AbstractMatrix* C) {
	double proportion = sqrt(GPUMatrix::SMThreads / (y * B->y)); //calculate a proportion to devide up the threads,
								//since y*B->x may be less than threads guard used in kernel
	dim3 Thread(std::max(proportion * y, 1.0), std::max(proportion * B->y, 1.0), 1); //allocate proportions of threads
	dim3 Block(ceil((double)y / Thread.x), ceil((double)B->y / Thread.y), 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUMultKernelB << < Block, Thread , 0, stream>> > (arr, B->arr, C->arr, C->y, C->x, x);
	//cudaStreamSynchronize(stream);
}

GPUMatrix* GPUMatrix::multiplyAB(AbstractMatrix* B) {  //multiplies two matrices on the GPU where the first of which is effectively transposed before carrayying out the calculation
	GPUMatrix* C = new GPUMatrix(x, B->y);

	if (y != B->x) { //check that the matrices are correct size for multiplying
		throw "Cannot multiply these"; //throw exception if they are
	}
	multiplyAB(B, C);
	return C;
}
void GPUMatrix::multiplyAB(AbstractMatrix* B, AbstractMatrix* C) {
	double proportion = sqrt(GPUMatrix::SMThreads / (x * B->y)); //calculate a proportion to devide up the threads,
								//since y*B->x may be less than threads guard used in kernel
	dim3 Thread(std::max(proportion * x, 1.0), std::max(proportion * B->y, 1.0), 1); //allocate proportions of threads
	dim3 Block(ceil((double)x / Thread.x), ceil((double)B->y / Thread.y), 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUMultKernelAB << < Block, Thread , 0, stream>> > (arr, B->arr, C->arr, C->y, C->x, y);
	//cudaStreamSynchronize(stream);
}

GPUMatrix* GPUMatrix::multiplyElementWise(AbstractMatrix* B) {
	GPUMatrix* C = new GPUMatrix(y, x);
	multiplyElementWise(B, C);
	return C;
}

void GPUMatrix::multiplyElementWise(AbstractMatrix* B, AbstractMatrix* C) {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUElementWiseMultiply << < Block, Thread , 0, stream>> > (arr, B->arr, C->arr, size);
	//cudaStreamSynchronize(stream);
}

GPUMatrix* GPUMatrix::divideElementWise(AbstractMatrix* B) {
	GPUMatrix* C = new GPUMatrix(y, x);
	divideElementWise(B, C);
	return C;
}

void GPUMatrix::divideElementWise(AbstractMatrix* B, AbstractMatrix* C) {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUElementWiseDivide << < Block, Thread, 0, stream >> > (arr, B->arr, C->arr, size);
	//cudaStreamSynchronize(stream);
}

GPUMatrix* GPUMatrix::sigmoid() {
	GPUMatrix* C = new GPUMatrix(y, x);
	sigmoid(C);
	return C;
}
void GPUMatrix::sigmoid(AbstractMatrix* C) {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUSigmoid << < Block, Thread , 0, stream>> > (arr, C->arr, size);
	//cudaStreamSynchronize(stream);
}

GPUMatrix* GPUMatrix::sigmoidDifferential() {
	GPUMatrix* C = new GPUMatrix(y, x);
	sigmoidDifferential(C);
	return C;
}
void GPUMatrix::sigmoidDifferential(AbstractMatrix* C) {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUSigmoidDifferential << < Block, Thread , 0, stream>> > (arr, C->arr, size);
	//cudaStreamSynchronize(stream);
}

GPUMatrix* GPUMatrix::add(AbstractMatrix* B) {
	GPUMatrix* C = new GPUMatrix(y, x);
	add(B, C);
	return C;
}
void GPUMatrix::add(AbstractMatrix* B, AbstractMatrix* C) {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUElementwiseAdd << < Block, Thread , 0, stream>> > (arr, B->arr, C->arr, size);
	//cudaStreamSynchronize(stream);
}

void GPUMatrix::addAssign(AbstractMatrix* B) {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUAddAssign << < Block, Thread , 0, stream>> > (arr, B->arr, size);
	//cudaStreamSynchronize(stream);
}

GPUMatrix* GPUMatrix::addConst(double B) {
	GPUMatrix* C = new GPUMatrix(y, x);
	addConst(B, C);
	return C;
}
void GPUMatrix::addConst(double B, AbstractMatrix* C) {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUAddConst << < Block, Thread , 0, stream>> > (arr, B, C->arr, size);
	//cudaStreamSynchronize(stream);
}

GPUMatrix* GPUMatrix::subtract(AbstractMatrix* B) {
	GPUMatrix* C = new GPUMatrix(y, x);
	subtract(B, C);
	return C;
}
void GPUMatrix::subtract(AbstractMatrix* B, AbstractMatrix* C) {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUElementwiseSubtract << < Block, Thread , 0, stream>> > (arr, B->arr, C->arr, size);
	//cudaStreamSynchronize(stream);
}

void GPUMatrix::subtractAssign(AbstractMatrix* B) {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUSubtractAssign << < Block, Thread , 0, stream>> > (arr, B->arr, size);
	//cudaStreamSynchronize(stream);
}



GPUMatrix* GPUMatrix::scale(double B) {
	GPUMatrix* C = new GPUMatrix(y, x);
	scale(B, C);
	return C;
}
void GPUMatrix::scale(double B, AbstractMatrix* C) {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUScale << < Block, Thread , 0, stream>> > (arr, B, C->arr, size);
	//cudaStreamSynchronize(stream);
}

double GPUMatrix::sum() const {
	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double)size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index
	double* totals;
	cudaMalloc(&totals, x*sizeof(double));
	GPUSum << < Block, Thread , 0, stream>> > (arr, x, y, totals);
	double* results;
	results = (double*)malloc(x*sizeof(double));
	cudaMemcpy(results, totals, x*sizeof(double), cudaMemcpyDeviceToHost);
	double total = 0;
	for(int i = 0; i < x; i++){
		total += results[i];
	}
	return total;
}


void GPUMatrix::convolute(AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* out, int outY, int outX, int outZ, int convY, int convX, int convZ) {
	

	/* double proportion = std::cbrt(GPUMatrix::SMThreads / (outX * outY * outZ)); //calculate a proportion to devide up the threads,
								//since y*B->x may be less than threads guard used in kernel
	dim3 Thread(std::max(proportion * outX, 1.0), std::max(proportion * outY, 1.0), std::max(proportion * outZ, 1.0)); //allocate proportions of threads
	dim3 Block(ceil((double)outX / Thread.x), ceil((double)outY / Thread.y), ceil((double)outZ / Thread.z));//if we need any more use threads for them, ceil ensures we never miss an index

	ConvKernel<< < Block, Thread, 0, stream >> > (arr, layer->arr, bias->arr, net->arr, out->arr, inY,  inZ,  outX, outY, outZ, convY, convX);//TODO REFACTOR FOR NET
	 *///cudaStreamSynchronize(stream);

}

void GPUMatrix::convBackprop(AbstractMatrix* in, AbstractMatrix* layer, AbstractMatrix* this_layer_conv_error, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* out, AbstractMatrix* out_error, AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR){	//prevError->fill(0);
	cudaMemset2DAsync(prevError->arr, sizeof(double), 0, prevError->x, prevError->y);
	//cudaDeviceSynchronize();
	//Matrix gradient(net->y, net->x);

	/* dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double) net->size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index
	int s = net->size;
	convBackpropErrorsKernel << < Block, Thread, 0, stream >> > (gradient->arr, net->arr, arr, s);
	double proportion = std::cbrt(GPUMatrix::SMThreads / (outX * outY * outZ)); //calculate a proportion to devide up the threads,
								//since y*B->x may be less than threads guard used in kernel
	dim3 Thread2(std::max(proportion * outX, 1.0), std::max(proportion * outY, 1.0), std::max(proportion * outZ, 1.0)); //allocate proportions of threads
	dim3 Block2(ceil((double)outX / Thread2.x), ceil((double)outY / Thread2.y), ceil((double)outZ / Thread2.z));//if we need any more use threads for them, ceil ensures we never miss an index

	convBackpropKernel << < Block2, Thread2, 0, stream >> > (arr, in->arr, layer->arr, prevError->arr, bias->arr, net->arr, outY, outX, outZ, convY, convX, convZ, LR, gradient->arr, 1 / (convX * convY), in->y);
 */
}
/* void GPUMatrix::convBackprop(AbstractMatrix* in, AbstractMatrix* layer, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* net, AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) {
	//prevError->fill(0);
	cudaMemset2DAsync(prevError->arr, sizeof(double), 0, prevError->x, prevError->y);
	//cudaDeviceSynchronize();
	//Matrix gradient(net->y, net->x);

	dim3 Thread(GPUMatrix::SMThreads, 1, 1); //allocate proportions of threads
	dim3 Block(std::ceil((double) net->size / GPUMatrix::SMThreads), 1, 1);//if we need any more use threads for them, ceil ensures we never miss an index
	int s = net->size;
	convBackpropErrorsKernel << < Block, Thread, 0, stream >> > (gradient->arr, net->arr, arr, s);
	double proportion = std::cbrt(GPUMatrix::SMThreads / (outX * outY * outZ)); //calculate a proportion to devide up the threads,
								//since y*B->x may be less than threads guard used in kernel
	dim3 Thread2(std::max(proportion * outX, 1.0), std::max(proportion * outY, 1.0), std::max(proportion * outZ, 1.0)); //allocate proportions of threads
	dim3 Block2(ceil((double)outX / Thread2.x), ceil((double)outY / Thread2.y), ceil((double)outZ / Thread2.z));//if we need any more use threads for them, ceil ensures we never miss an index

	convBackpropKernel << < Block2, Thread2, 0, stream >> > (arr, in->arr, layer->arr, prevError->arr, bias->arr, net->arr, outY, outX, outZ, convY, convX, convZ, LR, gradient->arr, 1 / (convX * convY), in->y);

} */



void GPUMatrix::randomFill(double min, double max) { //fills the matrix with a random values in the desired range
	std::uniform_real_distribution<double> RandDist(min, max);
	double* Temporaryarray;
	Temporaryarray = (double*)malloc(sizeof(double) * size);
	for (int index = 0; index < size; index++) {
		Temporaryarray[index] = RandDist(Matrix::mt);
	}
	cudaMemcpy(arr, Temporaryarray, sizeof(double) * size, cudaMemcpyHostToDevice);
	free(Temporaryarray);

}

void GPUMatrix::randomFill(double negmin, double negmax, double min, double max) { //fills the matrix with random values in the given ranges, used to exclude 0's from the range required for weights and biases
	std::uniform_real_distribution<double> SignRandDist(0, 2);
	std::uniform_real_distribution<double> RandDist(min, max);
	std::uniform_real_distribution<double> RandNegDist(negmin, negmax);
	double* Temporaryarray;
	Temporaryarray = (double*)malloc(sizeof(double) * size);
	for (int index = 0; index < size; index++) {
		if (SignRandDist(Matrix::mt) > 1) {
			Temporaryarray[index] = RandDist(Matrix::mt);
		}
		else {
			Temporaryarray[index] = RandNegDist(Matrix::mt);
		}
	}
	cudaMemcpy(arr, Temporaryarray, sizeof(double) * size, cudaMemcpyHostToDevice);
	free(Temporaryarray);
}

void GPUMatrix::transpose(GPUMatrix* B) {
	double proportion = sqrt(GPUMatrix::SMThreads / (x * y)); //calculate a proportion to devide up the threads,
								//since y*B->x may be less than threads guard used in kernel
	dim3 Thread(std::max(proportion * x, 1.0), std::max(proportion * y, 1.0), 1); //allocate proportions of threads
	dim3 Block(ceil((double)x / Thread.x), ceil((double)y / Thread.y), 1);//if we need any more use threads for them, ceil ensures we never miss an index

	GPUTranspose << < Block, Thread , 0, stream>> > (arr, B->arr, y, x);
	//cudaStreamSynchronize(stream);
}

void GPUMatrix::print() const { //prints the matrix to the console
	CPUMatrix temp(this);
	temp.print();
}

