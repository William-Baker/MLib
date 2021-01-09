#pragma once
#include "Matrix.hpp"
#include <future>






class Layer {
	Layer* nextLayer;
	Layer* prevLayer;
public:
	
	enum LayerType
	{
		FF,
		CV,
		REC,
		INPUT,
		OUTPUT
	};
	LayerType type;
	
	Layer(LayerType typeIn){
		type = typeIn;
		nextLayer = 0;
		prevLayer = 0;
	}

	virtual void compute() = 0;
	virtual void compute(MLStruct<double>* in) = 0;
	virtual void backprop(double LR) = 0;
	virtual void backprop(MLStruct<double>* err, double LR) = 0;
	virtual void randomise() = 0;
	virtual MLStruct<double>* getInputError() = 0;
	virtual MLStruct<double>* getOutput() = 0;
	virtual void print() = 0;


	inline void setNextLayer(Layer* layer) {
		nextLayer = layer;
		if(layer) layer->prevLayer = this;
	}
	inline void setPrevLayer(Layer* layer) {
		prevLayer = layer;
		if(layer) layer->nextLayer = this;
	}

	

	inline Layer* getNextLayer() {return nextLayer;}
	inline Layer* getPrevLayer() {return prevLayer;}
};

class InputLayer : public Layer {
public:
	MLStruct<double>* output;

	// InputLayer& operator=(InputLayer& m) {
	// 	std::swap(this->output, m.output);
	// 	std::swap(this->nextLayer, m.nextLayer);
	// 	std::swap(this->prevLayer, m.prevLayer);
	// 	std::swap(this->type, m.type);
	// 	return *this;
	// }

	InputLayer() : Layer(INPUT){
	}

	InputLayer(Matrix* output) : InputLayer() {
		this->output = (MLStruct<double>*)output;
	}

	InputLayer(MLStruct<double>* output) : InputLayer() {
		this->output = output;
	}

	/**
	 * Just passes the computation forward to the next element
	 */
	void compute() override {
		if (getNextLayer()) {
			getNextLayer()->compute();
		}
	}

	/**
	 * Just passes the computation forward to the next element, without modifying the input
	 */
	void compute(MLStruct<double>* input) override {
		output = input; //Set the output as the input directly as this layer does nothing
		if (getNextLayer()) {
			getNextLayer()->compute(input);
		}
	}

	void backprop(MLStruct<double>* err, double LR) override {
		return;
	}

	void backprop(double LR) override {
		return;
	}

	void randomise() override {
		return;
	}

	MLStruct<double>* getInputError() override {
		return nullptr;
	}

	MLStruct<double>* getOutput() override {
		return output;
	}

	void print() override{
		std::cout << "--------------- Input Layer ---------------" << std::endl;
		output->print();
	}

};
//TODO add warnings for empty functions
class OutputLayer : public Layer{
	public:
	OutputLayer() : Layer(OUTPUT){
	
	}

	void compute() override {
	}

	void compute(MLStruct<double>* input) {
	}

	virtual MLStruct<double>* computeErrorDifferential(MLStruct<double>* target) = 0;
	virtual MLStruct<double>* computeError(MLStruct<double>* target) = 0;

	/**
	 * Pass backprop to previous element without modification
	 */
	void backprop(MLStruct<double>* err, double LR) override {
		getPrevLayer()->backprop(err, LR);
	}

	/**
	 * Pass backprop to previous element without modification
	 */
	void backprop(double LR) override {
		getPrevLayer()->backprop(LR);
	}


	void randomise() override {
		return;
	}

	MLStruct<double>* getOutput() override {
		return getPrevLayer()->getOutput();
	}

};

class ErrorHalfSquaredOutputLayer : public OutputLayer{
	Matrix error;
	Matrix intermediate;
	public:
	ErrorHalfSquaredOutputLayer(size_t input_size) : OutputLayer(), error(input_size, 1), intermediate(input_size, 1) {}
	MLStruct<double>* computeErrorDifferential(MLStruct<double>* target) override{
		Matrix prev_out(getPrevLayer()->getOutput(), false); //create temporary wrapper matrix for prev layer output matrix
		Matrix target_matrix(target, false); //create temporary wrapper matrix for target matrix
		target_matrix.subtract(prev_out, error);
		return error.getStrategy();
	}
	MLStruct<double>* computeError(MLStruct<double>* target) {
		Matrix prev_out(getPrevLayer()->getOutput(), false); //create temporary wrapper matrix for prev layer output matrix
		Matrix target_matrix(target, false); //create temporary wrapper matrix for target matrix
		target_matrix.subtract(prev_out, error);
		error.multiply(error, intermediate);
		intermediate.scale(0.5, error);
		return error.getStrategy();
	}
	MLStruct<double>* getInputError() override {return error.getStrategy();}
	void print() override{
		std::cout << "--------------- Output Error ---------------" << std::endl;
		error.print();
	}
};

// class ErrorFunction {
// public:
// 	virtual double Error(double classification, double target) = 0;
// 	virtual void Error(Matrix& classification, Matrix& target, Matrix& C) = 0;
// 	virtual double gradient(double classification, double target) = 0;
// 	virtual void gradient(Matrix& classification, Matrix& target, Matrix& C) = 0;
// };
// class ErrorHalfSquared : public ErrorFunction {
// 	Matrix temp1;
// 	Matrix temp2;
// 	bool used = false;
// 	double Error(double classification, double target) override {
// 		return (0.5 * ((classification - target) * (classification - target)));
// 	}
// 	void Error(Matrix& classification, Matrix& target, Matrix& C) override{
// 		if(!used || (classification.width() != temp1.width() || classification.height() != temp1.height())){
// 			temp1 = Matrix(classification.height(),classification.width());
// 			temp2 = Matrix(classification.height(),classification.width());
// 			used = true;
// 		}
// 		classification.subtract(target,temp1 );
// 		classification.subtract(target, temp2 );
// 		temp1.multiplyElementWise(temp2, C);
// 		C.scale(0.5, C);
// 	}
// 	double gradient(double classification, double target) override {
// 		return target - classification;
// 	}
// 	void gradient(Matrix& classification, Matrix& target, Matrix& C) override {
// 		target.subtract(classification, C);	
// 	}
// };
// class ErrorAsym : public ErrorFunction {
// 	Matrix x;
// 	Matrix neg;
// 	Matrix op;
// 	bool used = false;
// 	double Error(double classification, double target) override {
// 		return (0.5 * ((target - classification) * (target - classification)));
// 	}
// 	double gradient(double classification, double target) override {
// 		double x = target - classification;
// 		return x / (1.1 - x);
// 	}
// 	void gradient(Matrix& classification, Matrix& target, Matrix& C) override {
// 		if(!used || (classification.width() != x.width() || classification.height() != x.height())){
// 			x = Matrix(classification.height(),classification.width());
// 			neg = Matrix(classification.height(),classification.width());
// 			op = Matrix(classification.height(),classification.width());
// 			used = true;
// 		}
// 		target.subtract(classification, x);
// 		x.scale(-1, neg);
// 		neg.addConst(1.1, op);
// 		x.divideElementWise(op, C);
// 	}
// };

class FeedForwardLayer : public Layer {
public:
	Matrix weights;
	Matrix biases;
	Matrix output;
	Matrix net;
	Matrix errorSig; //The error signal at the input
	Matrix temp2;
	Matrix gradient;
	Matrix weightsDeltas;
	Matrix* input;
	
	
	/* FeedForwardLayer& operator=(FeedForwardLayer&& m){
		std::swap(this->weights, m.weights);
		std::swap(this->biases, m.biases);
		std::swap(this->output, m.output);
		std::swap(this->net , m.net);
		std::swap(this->errorSig , m.errorSig);
		std::swap(this->temp2 , m.temp2);
		std::swap(this->gradient , m.gradient);
		std::swap(this->weightsDeltas , m.weightsDeltas);
		std::swap(this->nextLayer , m.nextLayer);
		std::swap(this->prevLayer , m.prevLayer);
		std::swap(this->type , m.type);

		return *this;
	} */
	FeedForwardLayer& operator=(FeedForwardLayer&& m) = default;

	FeedForwardLayer() : Layer(FF){}
	FeedForwardLayer(size_t inputs, size_t outputs, Layer* prevLayer) : FeedForwardLayer() {
		weights = Matrix(outputs, inputs);
		weightsDeltas = Matrix(outputs, inputs);
		biases = Matrix(outputs, 1);
		output = Matrix(outputs, 1);
		errorSig = Matrix(inputs, 1);
		net = Matrix(outputs, 1);
		temp2 = Matrix(outputs, 1);
		gradient = Matrix(outputs, 1);

		setPrevLayer(prevLayer);
	}

	FeedForwardLayer(Matrix& Weights, Matrix& Biases, Layer* prevLayer) : FeedForwardLayer(){
		weights = std::move(Weights);
		weightsDeltas = Matrix(weights.height(), weights.width());
		biases = Matrix(weights.height(), 1);
		output = Matrix(weights.height(), 1);
		errorSig = Matrix(weights.width(), 1);
		net = Matrix(weights.height(), 1);
		temp2 = Matrix(weights.height(), 1);
		gradient = Matrix(weights.height(), 1);

		setPrevLayer(prevLayer);
	}






	virtual MLStruct<double>* getInputError() override {
		return errorSig.getStrategy();
	}
	virtual MLStruct<double>* getOutput() override {
		return output.getStrategy();
	}
	


//late oct



	void randomise() override {
		weights.randomFill(-0.3, -0.05, 0.05, 0.3);
		biases.randomFill(-0.3, -0.05, 0.05, 0.3);
		//weights.randomFill(0, 1);
		//biases.randomFill(0, 1);
	}

	
	// virtual void compute() override {
	// 	Matrix tmp(getPrevLayer()->getOutput(), false);
	// 	compute(tmp);
	// }

	void compute(MLStruct<double>* inputIn) override {
		delete input;
		input = new Matrix(inputIn, false);
		compute();
	}
	void compute() override {
		weights.multiply(*input, output);
		output.add(biases, net);
		net.sigmoid(output);
		if (getNextLayer() != 0) {
			getNextLayer()->compute();
		}
	}

	void calculateErrorSignal(Matrix& outputError) {
		weights.multiplyA(outputError, errorSig);
		/*std::cout << "------- Output Error ------------" << std::endl;
		outputError.print();
		std::cout << "------- Weights ------------" << std::endl;
		weights.print();
		std::cout << "------- Error Signal ------------" << std::endl;
		errorSig.print();
		std::cout << std::endl;
		std::cout << std::endl;
		*/
	}

	void update(double LR, Matrix& errorSigAtOutput) {
		output.sigmoidDifferential(gradient);
		gradient.scale(LR, temp2);
		temp2.multiplyElementWise(errorSigAtOutput, gradient);
		gradient.multiplyB(*input, weightsDeltas);
		weights.addAssign(weightsDeltas);
		//std::cout << "------- Weights Deltas ------------" << std::endl;
		//weightsDeltas.print();
		biases.addAssign(gradient);
	}

	void backprop(double LR) override {
		backprop(getNextLayer()->getInputError(), LR);
	}
	void backprop(MLStruct<double>* err, double LR) override{
		Matrix tmp(err,false);
		backprop(tmp, LR);
	}
	void backprop(Matrix& outErrorSig, double LR)  {
		calculateErrorSignal(outErrorSig);
		update(LR, outErrorSig);
		getPrevLayer()->backprop(LR);	
	}
	
	

	void print() override {
		std::cout << "--------------- FF Weights Layer Y: " << weights.height() << " X: " << weights.width() <<  " ---------------" << std::endl;
		weights.print();
		std::cout << "--------------- FF Weights Deltas Layer Y: " << weightsDeltas.height() << " X: " << weightsDeltas.width() << " ---------------" << std::endl;
		weightsDeltas.print();
		std::cout << "--------------- FF Biases Layer Y: " << biases.height() << " ---------------" << std::endl;
		biases.print();
		std::cout << "--------------- FF Output Layer Y: " << output.height() << " ---------------" << std::endl;
		output.print();
		std::cout << "--------------- FF Out Error Layer Y: " << errorSig.height() << " ---------------" << std::endl;
		errorSig.print();
		
	}


	
};

class ConvLayer : public Layer {
	
public:
	Matrix layer; //Convolution layer
	Matrix bias; //bias added to convolution output
	Matrix output; //the output of this layer
	Matrix layerError; //error in the convoltuiional layer
	//Matrix unactivated_output; //output of the layer before activation function -- Not needed as not used
	Matrix gradient; //gradient at the output
	Matrix errorSignal;//The Error Signal at the input to this layer
	Matrix* input;//Pointer to the last used input
	int inX;
	int inY;
	int inZ;

	int outX;
	int outY;
	int outZ;

	int convX;
	int convY;

	ConvLayer() : Layer(CV) {}

private:
	void allocate() {
		layer = Matrix(convY * inZ, convX * outZ);
		bias = Matrix(outZ, 1);
		output = Matrix(outY * outZ, outX);
		layerError = Matrix(convY * inZ, convX * outZ);
		gradient = Matrix(outY * outZ, outX);
		errorSignal = Matrix(inY * inZ, inX);
		
	}
public:

	ConvLayer(int inX, int inY, int inZ, int outZ, int convX, int convY, Layer* prevLayer) : ConvLayer() {
		this->inX = inX;
		this->inY = inY;
		this->inZ = inZ;

		this->outX = inX - convX + 1;
		this->outY = inY - convY + 1;
		this->outZ = outZ;

		this->convX = convX;
		this->convY = convY;

		setPrevLayer(prevLayer);

		allocate();
	}

	ConvLayer(ConvLayer& m) = delete;




	virtual MLStruct<double>* getInputError() override {
		return errorSignal.getStrategy();
	}
	virtual MLStruct<double>* getOutput() override {
		return output.getStrategy();
	}

	void randomise() override {
		layer.randomFill(-0.3, -0.05, 0.05, 0.3);
		bias.randomFill(-0.3, -0.05, 0.05, 0.3);

	}
	

	void compute(MLStruct<double>* inputIn) override {
		delete input;
		input = new Matrix(inputIn, false);
		compute();
	}

	void compute() override {
		//AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* unactivated_output, AbstractMatrix* out
		input->convolute(layer,  bias, output, outY, outX, outZ, convY, convX, inZ);
		if (getNextLayer()) {
			getNextLayer()->compute();
		}
	}
	// void compute() override {
	// 	compute(getPrevLayer()->getOutput());
	// }


	void backprop(Matrix &outError, double LR) {	
		outError.convBackprop(*input, layer, layerError, errorSignal, bias, output, outError, gradient, outY, outX, outZ, convY, convX, inZ, LR);

		if (getPrevLayer()) {
			getPrevLayer()->backprop(LR);
		}

	}
	void backprop(double LR) override {
		backprop(getNextLayer()->getInputError(), LR);
	}
	void backprop(MLStruct<double>* err, double LR) override {
		Matrix tmp(err, false);
		backprop(tmp, LR);
	}
	
	//Matrix layer;
	//Matrix bias;
	//Matrix output;
	//Matrix net;
	//Matrix errorSignal;//The Error Signal at teh input
	void print() override {
		std::cout << "--------------- CV Conv Layer Y: " << convY << " X: " << convX << " Z: " << inZ << "---------------" << std::endl;
		layer.print();
		std::cout << "--------------- CV Bias Layer Y: " << bias.height() << " ---------------" << std::endl;
		bias.print();
		std::cout << "--------------- CV output Layer Y: " << outY << " X: " << outX << " Z: " << outZ << " ---------------" << std::endl;
		output.print();
		std::cout << "--------------- CV error Layer Y: " << outY << " X: " << outX << " Z: " << outZ << " ---------------" << std::endl;
		errorSignal.print();
	}

	~ConvLayer(){
		delete input;
	}


};



//TODO refactor how layers are added
class NeuralNetwork {
	
public:
	std::vector<Layer*> layers;
	InputLayer input_layer;
 	OutputLayer* output_layer;

	enum ErrorFunction{
		Default, ErrorHalfSquared
	};
	ErrorFunction error_function;

	NeuralNetwork(ErrorFunction err = Default) {
		error_function = err;
		output_layer = 0;
	}
private:
	void generate_output_layer(){
		size_t output_layer_size = layers.back()->getOutput()->get_size();
		if((error_function == Default) || (error_function == ErrorHalfSquared)){
			output_layer = new ErrorHalfSquaredOutputLayer(output_layer_size);
		}
		output_layer->setPrevLayer(finalLayer());
	}
	
	void link_layers(Layer* left, Layer* right){
		if(left->getOutput()){
			if(dynamic_cast<FeedForwardLayer*>(right)){
				auto r = static_cast<FeedForwardLayer*>(right);
				r->input = new Matrix(Matrix(left->getOutput(), false).copy_keeping_same_data());
				r->input->flatten();
				//TODO add verifaction here
			}
			else if (dynamic_cast<ConvLayer*>(right)){
				auto r = static_cast<ConvLayer*>(right);
				r->input = new Matrix(left->getOutput(), false);
				//TODO verify?
			}
		}
	}

public:

	NeuralNetwork(std::vector<Layer*> layers, ErrorFunction err = Default) : NeuralNetwork() {
		error_function = err;
		this->layers = layers;
		Layer* prevLayer = &input_layer;
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->setPrevLayer(prevLayer);
			link_layers(prevLayer, layers[i]);
			prevLayer = layers[i];
		}
		generate_output_layer();
	}

	void addLayer(Layer* layer) {
		layers.push_back(layer);
		if (layers.size() == 1) {
			layer->setPrevLayer(&input_layer);
		}
		else {
			layer->setPrevLayer(layers[layers.size() - 2]);
			link_layers(layers[layers.size() - 2], layers[layers.size() - 1]);
		}
		
		delete output_layer; //delete the old output layer if applicable
		generate_output_layer();
	}

	

	void randomise() {
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->randomise();
		}
	}

	void compute(MLStruct<double>* inputs) {
		input_layer.compute(inputs);
	}

	/**
	 * TODO refactor for generic
	 * @param target a Matrix of target values, of equal dimension to the output
	 * @return the performance of the network 0-1
	 */
	double backprop(Matrix& target, double LR = 0.05) {

		Matrix output_error(output_layer->computeErrorDifferential(target.getStrategy()), false);

		finalLayer()->backprop(output_error.getStrategy(), LR);

		return output_error.sum();

	}

	double get_error(Matrix& target) {
		return Matrix(output_layer->computeError(target.getStrategy()), false).sum();
	}

	Layer* finalLayer() {
		return layers[layers.size() - 1];
	}

	Matrix get_output(){
		return Matrix(finalLayer()->getOutput(), false);
	}

	void print() {
		input_layer.print();
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->print();
		}
	}

	~NeuralNetwork(){
		delete output_layer;
	}
};

class MLCase {
public:
	MLStruct<double>* inputs;
	MLStruct<double>* outputs;
	//std::mutex resMutex;

	//TODO add mutex to MLCase

	MLCase(){};

	/**
	 * Overriden copy constructor as resMutex has no copy constructor
	 */
	// MLCase(MLCase& x){

	// }

	MLCase(MLStruct<double>* in, MLStruct<double>* out){
		inputs = in;
		outputs = out;
	}

	/**
	 * swaps a GPU case with a CPU case
	 * @param gpu an MLCase stored on the GPU
	 * @param cpu an MLCase stored on the CPU
	 */
	static void swap(MLCase* gpu, MLCase* cpu) {
		if(dynamic_cast<GPUMatrix*>(gpu->inputs)){
			//std::lock_guard<std::mutex> lockg(gpu->resMutex);
			//std::lock_guard<std::mutex> lockc(cpu->resMutex);
			
			auto new_pair = swapMatrix(dynamic_cast<GPUMatrix*>(gpu->inputs), dynamic_cast<CPUMatrix*>(cpu->inputs));
			gpu->inputs = new_pair.first;
			cpu->inputs = new_pair.second;
			new_pair = swapMatrix(dynamic_cast<GPUMatrix*>(gpu->outputs), dynamic_cast<CPUMatrix*>(cpu->outputs));
			gpu->outputs = new_pair.first;
			cpu->outputs = new_pair.second;
		}
		else{
			throw(std::invalid_argument("gpu case not a GPU matrix"));
		}
	}



private:

	static std::pair<GPUMatrix*, CPUMatrix*> swapMatrix(GPUMatrix* g, CPUMatrix* c) {
		
		if(g || c == 0) throw(std::invalid_argument("null pointer arguments"));

		CPUMatrix* CPUInputsTemp = new CPUMatrix(g);
		g->~GPUMatrix();

		g = new GPUMatrix(c);

		c->~CPUMatrix();

		c = CPUInputsTemp;

		return {g, c};
	}

};


