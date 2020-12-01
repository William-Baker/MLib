#pragma once
#include "Matrix.hpp"
#include <future>

class Layer {
public:
	Layer* nextLayer = 0;
	Layer* prevLayer = 0;
	enum LayerType
	{
		FF,
		CV,
		REC,
		INPUT
	};
	LayerType type;
	
	virtual void compute() = 0;
	virtual void backprop(double LR) = 0;
	virtual void backprop(MLStruct<double>* err, double LR) = 0;
	virtual void randomise() = 0;
	virtual MLStruct<double>* getInputError() = 0;
	virtual MLStruct<double>* getOutput() = 0;
	virtual void print() = 0;

	void setNextLayer(Layer* layer) {
		nextLayer = layer;
		if(layer) layer->prevLayer = this;
	}
	void setPrevLayer(Layer* layer) {
		prevLayer = layer;
		if(layer) layer->nextLayer = this;
	}

	Layer* getNextLayer() {return nextLayer;}
	Layer* getPrevLayer() {return prevLayer;}
};

class InputLayer : public Layer {
public:
	MLStruct<double>* output;

	InputLayer& operator=(InputLayer& m) {
		std::swap(this->output, m.output);
		std::swap(this->nextLayer, m.nextLayer);
		std::swap(this->prevLayer, m.prevLayer);
		std::swap(this->type, m.type);
		return *this;
	}

	void common() {
		type = Layer::LayerType::INPUT;
		prevLayer = 0;
		nextLayer = 0;
	}

	InputLayer() {

	}

	InputLayer(Matrix* output) {
		this->output = (MLStruct<double>*)output;
	}

	InputLayer(MLStruct<double>* output) {
		this->output = output;
	}

	void compute() override {
		if (nextLayer) {
			nextLayer->compute();
		}
	}

	void compute(MLStruct<double>* input) {
		output = input;
		if (nextLayer) {
			nextLayer->compute();
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
		return 0;
	}

	MLStruct<double>* getOutput() override {
		return output;
	}

	void print() override{
		std::cout << "--------------- Input Layer ---------------" << std::endl;
		output->print();
	}
};

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
	MLStruct<double>* input;
	
	
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

	void common() {
		type = Layer::LayerType::FF;
		prevLayer = 0;
		nextLayer = 0;
	}
	FeedForwardLayer(){
		common();
	}
	FeedForwardLayer(size_t inputs, size_t outputs, Layer* prevLayer) {
		common();
		weights = Matrix(outputs, inputs);
		weightsDeltas = Matrix(outputs, inputs);
		biases = Matrix(outputs, 1);
		output = Matrix(outputs, 1);
		errorSig = Matrix(inputs, 1);
		net = Matrix(outputs, 1);
		temp2 = Matrix(outputs, 1);
		gradient = Matrix(outputs, 1);
		this->prevLayer = prevLayer;
		setPrevLayer(prevLayer);
		
	}

	FeedForwardLayer(Matrix& Weights, Matrix& Biases, Layer* prevLayer) {
		common();
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

	
	virtual void compute() override {
		Matrix tmp(getPrevLayer()->getOutput(), false);
		compute(tmp);
	}

	void compute(MLStruct<double>* input)  {
		Matrix tmp(input, false);
		compute(tmp);
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
		Matrix input_wrapper(input, false);
		gradient.multiplyB(input_wrapper, weightsDeltas);
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

	private:
		void compute(Matrix& input) {
			this->input = input.getStrategy();
			weights.multiply(input, output);
			output.add(biases, net);
			net.sigmoid(output);
			if (getNextLayer() != 0) {
				getNextLayer()->compute();
			}
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

	void common() {
		type = Layer::LayerType::CV;
		prevLayer = 0;
		nextLayer = 0;
	}

	ConvLayer() {
		common();
	}

	ConvLayer(int inX, int inY, int inZ, int outZ, int convX, int convY, Layer* prevLayer) {
		common();
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


	ConvLayer& operator=(ConvLayer& m) {
		//TODO Check if nextLayer and prevLayer get moved

		this->layer = std::move(m.layer);
		this->bias = std::move(m.bias);
		this->output = std::move(m.output);
		this->layerError = std::move(m.layerError);
		//this->unactivated_output = std::move(m.unactivated_output);
		this->gradient = std::move(m.gradient);
		this->errorSignal = std::move(m.errorSignal);
		this->input = std::move(m.input);
		this->nextLayer = std::move(m.nextLayer);
		this->prevLayer = std::move(m.prevLayer);
		this->type = std::move(m.type);

		//std::swap(this->layer, m.layer);
		//std::swap(this->bias, m.bias);
		//std::swap(this->output, m.output);
		//std::swap(this->net, m.net);
		//std::swap(this->errorSignal, m.errorSignal);
		//std::swap(this->gradient, m.gradient);
		//std::swap(this->nextLayer, m.nextLayer);
		//std::swap(this->prevLayer, m.prevLayer);
		//std::swap(this->type, m.type);
		this->inX = m.inX;
		this->inY = m.inY;
		this->inZ = m.inZ;

		this->outX = m.outX;
		this->outY = m.outY;
		this->outZ = m.outZ;

		this->convX = m.convX;
		this->convY = m.convY;

		

		return *this;
	}




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
	



	void compute(Matrix& input) {
		//AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* unactivated_output, AbstractMatrix* out
		this->input = &input;
		input.convolute(layer,  bias, output, outY, outX, outZ, convY, convX, inZ);
		if (getNextLayer()) {
			getNextLayer()->compute();
		}
	}
	void compute() override {
		compute(prevLayer->getOutput());
	}
	void compute(MLStruct<double>* input){
		compute(input);
	}

	void backprop(Matrix &outError, double LR) {
		/*std::cout << "-------------- Error -------------" << std::endl;
		outError.print();
		std::cout << "-------------- Inputs --------" << std::endl;
		((Matrix*)((Matrix*)prevLayer->getOutput()))->print();
		std::cout << "-------------- Filter Pre --------" << std::endl;
		layer.print();*/
//AbstractMatrix* in, AbstractMatrix* layer, AbstractMatrix* this_layer_conv_error, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* out, AbstractMatrix* out_error, AbstractMatrix* gradient
		
		outError.convBackprop(*input, layer, layerError, errorSignal, bias, output, outError, gradient, outY, outX, outZ, convY, convX, inZ, LR);
		

		/*std::cout << "-------------- Filter Post --------" << std::endl;
		layer.print();
		std::cout << "-------------- Error In--------" << std::endl;
		errorSignal.print();*/ 

		if (prevLayer) {
			prevLayer->backprop(LR);
		}

	}
	void backprop(double LR) override {
		backprop(nextLayer->getInputError(), LR);
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
		for (int i = 0; i < inZ; i++) {
			for (int j = 0; j < convX - 1; j++) {
				std::cout << "-";
			}
			std::cout << "�";
		}
		std::cout << std::endl;
		layer.print();
		std::cout << "--------------- CV Bias Layer Y: " << bias.height() << " ---------------" << std::endl;
		bias.print();
		std::cout << "--------------- CV output Layer Y: " << outY << " X: " << outX << " Z: " << outZ << " ---------------" << std::endl;
		output.print();
		std::cout << "--------------- CV error Layer Y: " << outY << " X: " << outX << " Z: " << outZ << " ---------------" << std::endl;
		errorSignal.print();
	}


private:
	void allocate() {
		layer = Matrix(convY * inZ, convX * outZ);
		bias = Matrix(outZ, 1);
		output = Matrix(outY * outZ, outX);
		layerError = Matrix(convY * inZ, convX * outZ);
		//unactivated_output = Matrix(outY * outZ, outX);		
		gradient = Matrix(outY * outZ, outX);
		errorSignal = Matrix(inY * inZ, inX);
		
	}
};


class ErrorFunction {
public:
	virtual double Error(double classification, double target) = 0;
	virtual void Error(Matrix& classification, Matrix& target, Matrix& C) = 0;
	virtual double gradient(double classification, double target) = 0;
	virtual void gradient(Matrix& classification, Matrix& target, Matrix& C) = 0;
};
class ErrorHalfSquared : public ErrorFunction {
	Matrix temp1;
	Matrix temp2;
	bool used = false;
	double Error(double classification, double target) override {
		return (0.5 * ((classification - target) * (classification - target)));
	}
	void Error(Matrix& classification, Matrix& target, Matrix& C) override{
		if(!used || (classification.width() != temp1.width() || classification.height() != temp1.height())){
			temp1 = Matrix(classification.height(),classification.width());
			temp2 = Matrix(classification.height(),classification.width());
			used = true;
		}
		classification.subtract(target,temp1 );
		classification.subtract(target, temp2 );
		temp1.multiplyElementWise(temp2, C);
		C.scale(0.5, C);

	}
	double gradient(double classification, double target) override {
		return target - classification;
	}
	void gradient(Matrix& classification, Matrix& target, Matrix& C) override {
		target.subtract(classification, C);
		
	}
};
class ErrorAsym : public ErrorFunction {
	Matrix x;
	Matrix neg;
	Matrix op;
	bool used = false;
	double Error(double classification, double target) override {
		return (0.5 * ((target - classification) * (target - classification)));
	}
	double gradient(double classification, double target) override {
		double x = target - classification;
		return x / (1.1 - x);
	}





	void gradient(Matrix& classification, Matrix& target, Matrix& C) override {
		if(!used || (classification.width() != x.width() || classification.height() != x.height())){
			x = Matrix(classification.height(),classification.width());
			neg = Matrix(classification.height(),classification.width());
			op = Matrix(classification.height(),classification.width());
			used = true;
		}
		target.subtract(classification, x);
		x.scale(-1, neg);
		neg.addConst(1.1, op);
		x.divideElementWise(op, C);
	}
};


class NeuralNetworkFF {
	
public:
	std::vector<FeedForwardLayer*> layers;
	InputLayer input;
	NeuralNetworkFF(std::vector<FeedForwardLayer*> layers) {
		this->layers = layers;
		layers[0]->setPrevLayer(&input);
		for (int i = 1; i < layers.size(); i++) {
			layers[i]->setPrevLayer(layers[i-1]);
		}
	}

	NeuralNetworkFF(std::vector<size_t> dimensions) {
		layers.resize(dimensions.size()-1);
		Layer* prevLayer = &input;
		for (int i = 0; i < layers.size(); i++) {
			FeedForwardLayer* a = new FeedForwardLayer(dimensions[i], dimensions[i + 1], prevLayer);

			layers[i] = a;
			layers[i]->randomise();
			prevLayer = layers[i];
		}
	}
	void compute(MLStruct<double>* inputs) {
		input.compute(inputs);
	}
	void backprop(Matrix& target, Matrix& outError, ErrorFunction* err) {

		err->gradient(finalLayer()->output, target, outError);
		std::cout << "o: " << finalLayer()->output.index(0) << " t: " << target.index(0) << " e:" << outError.index(0) << std::endl;
		
		finalLayer()->backprop(outError, 0.5);
		/*finalLayer().calculateErrorSignal(outError);
		for (int i = layers.size() - 2; i > 0; i--) {
			layers[i].calculateErrorSignal(layers[i + 1].errorSig);
		}


		int i = 0;
		for (; i < layers.size()-1; i++) {
			layers[i].update(LR, layers[i + 1].errorSig, *layers[i].input);
		}
		layers[i].update(LR, outError, *layers[i].input);*/
	}
	FeedForwardLayer* finalLayer() {
		return layers[layers.size() - 1];
	}
	void randomise(){
		for(int i = 0; i < layers.size(); i++){
			layers[i]->randomise();
		}
	}
	void print() {
		std::cout << "		Input:" << std::endl;
		((Matrix*)input.getOutput())->print();
		for (int i = 0; i < layers.size(); i++) {
			std::cout << "		Layer: " << i << std::endl;
			layers[i]->weights.print();
		}
		std::cout << "		Outputs" << std::endl;
		finalLayer()->output.print();
	}
};

class NeuralNetwork {
public:
	std::vector<Layer*> layers;
	InputLayer input;
	ErrorFunction* err;
	Matrix outError;
	
	NeuralNetwork(ErrorFunction* err) {
		this->err = err;
	}

	NeuralNetwork(std::vector<Layer*> layers, ErrorFunction* err) {
		this->err = err;
		this->layers = layers;
		Layer* prevLayer = &input;
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->setPrevLayer(prevLayer);
			prevLayer = layers[i];
		}
	}

	void addLayer(Layer* layer) {
		layers.push_back(layer);
		if (layers.size() == 1) {
			layer->setPrevLayer(&input);
		}
		else if(layer != NULL) {
			layer->setPrevLayer(layers[layers.size() - 2]);
		}
		outError = Matrix(finalLayer()->getOutput()->get_size(),1);
		
	}

	

	void randomise() {
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->randomise();
		}
	}

	void compute(MLStruct<double>* inputs) {
		input.compute(inputs);
	}

	/**
	 * @param target a Matrix of target values, of equal dimension to the output
	 * @return the performance of the network 0-1
	 */
	double backprop(Matrix& target, double LR = 0.05) {
		//Matrix output(finalLayer()->getOutput(), false);
		
		
		//err->gradient(output, target, outError);

		err->gradient(((FeedForwardLayer*)finalLayer())->output, target, outError);
		//std::cout << "o: " << ((FeedForwardLayer*)finalLayer())->output.index(0) << " t: " << target.index(0) << " e:" << outError.index(0) << std::endl;

		finalLayer()->backprop(outError.getStrategy(), LR);

		err->Error(dynamic_cast<FeedForwardLayer*>(finalLayer())->output, target, outError);
		
		return outError.sum();

	}

	Layer* finalLayer() {
		return layers[layers.size() - 1];
	}

	void print() {
		input.print();
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->print();
		}
	}
};

class MLCase {
public:
	MLStruct<double>* inputs;
	MLStruct<double>* outputs;
	static std::mutex resMutex;

	MLCase(){};

	MLCase(MLStruct<double>* in, MLStruct<double>* out){
		inputs = in;
		outputs = out;
	}

	static void swap(MLCase* left, MLCase* right) {
		try
		{
			GPUMatrix* GPUInputs = dynamic_cast<GPUMatrix*>(left->inputs);
			GPUMatrix* GPUOutputs = dynamic_cast<GPUMatrix*>(left->outputs);
			CPUMatrix* CPUInputs = dynamic_cast<CPUMatrix*>(right->inputs);
			CPUMatrix* CPUOutputs = dynamic_cast<CPUMatrix*>(right->outputs);
			swapMatrix(GPUInputs, GPUOutputs, CPUInputs, CPUOutputs);

			//left GPU, right CPU
		}
		catch (const std::exception&)
		{
			try {
				CPUMatrix* CPUInputs = dynamic_cast<CPUMatrix*>(left->inputs);
				CPUMatrix* CPUOutputs = dynamic_cast<CPUMatrix*>(left->outputs);
				GPUMatrix* GPUInputs = dynamic_cast<GPUMatrix*>(right->inputs);
				GPUMatrix* GPUOutputs = dynamic_cast<GPUMatrix*>(right->outputs);
				swapMatrix(GPUInputs, GPUOutputs, CPUInputs, CPUInputs);
				
				//left CPU, right GPU
			}
			catch (const std::exception&) {
				//Both Same
				std::swap(left->inputs, right->inputs);
				std::swap(left->outputs, right->outputs);

			}
		}
		std::cout << "unknown MLStruct" << std::endl;
	}
private:
	static void swapMatrix(GPUMatrix* GPUInputs, GPUMatrix* GPUOutputs, CPUMatrix* CPUInputs, CPUMatrix* CPUOutputs) {
		std::lock_guard<std::mutex> lock(resMutex);
		CPUMatrix* CPUInputsTemp = new CPUMatrix(GPUInputs->y, GPUInputs->x, GPUInputs->copy_to_CPU());
		CPUMatrix* CPUOutputsTemp = new CPUMatrix(GPUOutputs->y, GPUOutputs->x, GPUOutputs->copy_to_CPU());

		GPUInputs->~GPUMatrix();
		GPUOutputs->~GPUMatrix();

		GPUInputs = new GPUMatrix(CPUInputs->y, CPUInputs->x, CPUInputs->arr, GPUMatrix::MEM::CPU);
		GPUOutputs = new GPUMatrix(CPUInputs->y, CPUInputs->x, CPUOutputs->arr, GPUMatrix::MEM::CPU);

		CPUInputs->~CPUMatrix();
		CPUOutputs->~CPUMatrix();

		CPUInputs = CPUInputsTemp;
		CPUOutputs = CPUOutputsTemp;
		

	}

};


