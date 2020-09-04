#pragma once
#include "Matrix.hpp"
//#include "mnist//mnist_reader.hpp"
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
	
	virtual void compute() = 0;;
	virtual void backprop(double LR) = 0;;
	virtual void backprop(MLStruct<double>* err, double LR) = 0;;
	virtual void randomise() = 0;;
	virtual MLStruct<double>* getInputError() = 0;;
	virtual MLStruct<double>* getOutput() = 0;;
	virtual void print() = 0;;

	void setNextLayer(Layer* layer) {
		nextLayer = layer;
		layer->prevLayer = this;
	}
	void setPrevLayer(Layer* layer) {
		prevLayer = layer;
		layer->nextLayer = this;
	}
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
	void compute(Matrix* input) {
		output = (MLStruct<double>*)input;
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
		if (prevLayer) {
			this->prevLayer->nextLayer = this;
		}
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
		this->prevLayer = prevLayer;
		if (prevLayer) {
			this->prevLayer->nextLayer = this;
		}
		
	}






	virtual MLStruct<double>* getInputError() override {
		return (MLStruct<double>*)&errorSig;
	}
	virtual MLStruct<double>* getOutput() override {
		return (MLStruct<double>*)&output;
	}
	






	void randomise() override {
		weights.randomFill(-0.3, -0.05, 0.05, 0.3);
		biases.randomFill(-0.3, -0.05, 0.05, 0.3);
		//weights.randomFill(0, 1);
		//biases.randomFill(0, 1);
	}

	
	virtual void compute() override {
		compute(*(Matrix*)(prevLayer->getOutput()));
	}

	void compute(MLStruct<double>* input)  {
		compute(*(Matrix*)input);
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

	void update(double LR, Matrix& errorSigAtOutput, Matrix& inputs) {
		output.sigmoidDifferential(gradient);
		gradient.scale(LR, temp2);
		temp2.multiplyElementWise(errorSigAtOutput, gradient);
		gradient.multiplyB(inputs, weightsDeltas);
		weights.addAssign(weightsDeltas);
		//std::cout << "------- Weights Deltas ------------" << std::endl;
		//weightsDeltas.print();
		biases.addAssign(gradient);
	}

	void backprop(double LR) override {
		backprop(*(Matrix*)nextLayer->getInputError(), LR);
	}
	void backprop(Matrix& outErrorSig, double LR)  {
		calculateErrorSignal(outErrorSig);
		update(LR, outErrorSig, *(Matrix*)(prevLayer->getOutput()));
		prevLayer->backprop(LR);	
	}
	void backprop(MLStruct<double>* err, double LR) override {
		backprop(*(Matrix*)err, LR);
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
		void compute(Matrix& inputs) {
			weights.multiply(inputs, output);
			output.add(biases, net);
			net.sigmoid(output);
			if (nextLayer != 0) {
				nextLayer->compute();
			}
		}
	
};

class ConvLayer : public Layer {
	
public:
	Matrix layer;
	Matrix bias;
	Matrix output;
	Matrix net;
	Matrix gradient;
	Matrix errorSignal;//The Error Signal at teh input
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

		this->prevLayer = prevLayer;
		if (prevLayer) {
			this->prevLayer->nextLayer = this;
		}

		allocate();
	}


	ConvLayer& operator=(ConvLayer& m) {
		this->layer = std::move(m.layer);
		this->output = std::move(m.output);
		this->net = std::move(m.net);
		this->errorSignal = std::move(m.errorSignal);
		this->gradient = std::move(m.gradient);
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
		return (MLStruct<double>*) & errorSignal;
	}
	virtual MLStruct<double>* getOutput() override {
		return (MLStruct<double>*) & output;
	}

	void randomise() override {
		layer.randomFill(-0.3, -0.05, 0.05, 0.3);
		bias.randomFill(-0.3, -0.05, 0.05, 0.3);

	}
	



	void compute(Matrix& input) {
		input.convolute(layer, bias, net, output, inX, inY, inZ, outX, outY, outZ, convX, convY);
		if (nextLayer) {
			nextLayer->compute();
		}
	}
	void compute() override {
		compute(*(Matrix*)prevLayer->getOutput());
	}
	void compute(MLStruct<double>* input){
		compute(*(Matrix*)input);
	}

	void backprop(Matrix &outError, double LR) {
		/*std::cout << "-------------- Error -------------" << std::endl;
		outError.print();
		std::cout << "-------------- Inputs --------" << std::endl;
		((Matrix*)((Matrix*)prevLayer->getOutput()))->print();
		std::cout << "-------------- Filter Pre --------" << std::endl;
		layer.print();*/

		outError.convBackprop(*(Matrix*)prevLayer->getOutput(), layer, errorSignal, bias, net, gradient, outY, outX, outZ, convY, convX, inZ, LR);
		

		/*std::cout << "-------------- Filter Post --------" << std::endl;
		layer.print();
		std::cout << "-------------- Error In--------" << std::endl;
		errorSignal.print();*/

		if (prevLayer) {
			prevLayer->backprop(LR);
		}

	}
	void backprop(double LR) override {
		backprop(*(Matrix*)nextLayer->getInputError(), LR);
	}
	void backprop(MLStruct<double>* err, double LR) override {
		backprop(*(Matrix*)err, LR);
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
		output = Matrix(outY * outZ, outX);
		net = Matrix(outY * outZ, outX);
		gradient = Matrix(outY * outZ, outX);
		bias = Matrix(outZ, 1);
		errorSignal = Matrix(inY * inZ, inX);
		
	}
};


class ErrorFunction {
public:
	virtual double Error(double classification, double target) = 0;;
	virtual double gradient(double classification, double target) = 0;;
	virtual void gradient(Matrix& classification, Matrix& target, Matrix& C) = 0;;
};
class ErrorHalfSquared : public ErrorFunction {
	double Error(double classification, double target) override {
		return (-0.5 * ((classification - target) * (classification - target)));
	}
	double gradient(double classification, double target) override {
		return target - classification;
	}
	void gradient(Matrix& classification, Matrix& target, Matrix& C) override {
		target.subtract(classification, C);
		
	}
};
class ErrorAsym : public ErrorFunction {
	double Error(double classification, double target) override {
		return (0.5 * ((target - classification) * (target - classification)));
	}
	double gradient(double classification, double target) override {
		double x = target - classification;
		return x / (1.1 - x);
	}





	void gradient(Matrix& classification, Matrix& target, Matrix& C) override {
		Matrix x = target.subtract(classification);
		Matrix neg = x.scale(-1);
		Matrix op = neg.addConst(1.1);
		x.divideElementWise(op, C);
		//target.subtract(classification, C);

	}
};


class NeuralNetworkFF {
	
public:
	std::vector<FeedForwardLayer*> layers;
	InputLayer input;
	NeuralNetworkFF(std::vector<FeedForwardLayer*> layers) {
		this->layers = layers;
		Layer* prevLayer = &input;
		for (int i = 0; i < layers.size(); i++) {
			prevLayer = layers[i];
			if (i != 0) {
				prevLayer->nextLayer = layers[i];
			}
		}
	}
	NeuralNetworkFF(std::vector<size_t> dimensions) {
		layers.resize(dimensions.size()-1);
		Layer* prevLayer = &input;
		for (int i = 0; i < layers.size(); i++) {
			FeedForwardLayer* a = new FeedForwardLayer(dimensions[i], dimensions[i + 1], prevLayer);
		    //FeedForwardLayer b = layers.at(i);
			//b = a;
			layers[i] = a;
			layers[i]->randomise();
			prevLayer = layers[i];
			//b.randomise();
		}
	}
	void compute(Matrix& inputs) {
		input.compute(&inputs);
	}
	void backprop(Matrix& target, Matrix& outError, ErrorFunction* err) {
		std::cout << "o: " << finalLayer()->output.index(0) << " t: " << target.index(0) << std::endl;

		err->gradient(finalLayer()->output, target, outError);
		
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

	NeuralNetwork() {
		
	}

	NeuralNetwork(std::vector<Layer*> layers) {
		this->layers = layers;
		Layer* prevLayer = &input;
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->setPrevLayer(prevLayer);
			prevLayer = layers[i];
		}
	}

	void addLayer(Layer* layer) {
		layers.resize(layers.size() + 1);
		if (layers.size() == 1) {
			layers[0] = layer;
			layer->setPrevLayer(&input);
		}
		else {
			layer->setPrevLayer(layers[layers.size() - 2]);
			layers[layers.size() - 1] = layer;

		}
	}

	

	void randomise() {
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->randomise();
		}
	}

	void compute(Matrix& inputs) {
		input.compute(&inputs);
	}

	void backprop(Matrix& target, Matrix& outError, ErrorFunction* err) {
		Matrix* output = (Matrix*)finalLayer()->getOutput();
		//std::cout << "o: " << output->index(0) << " t: " << target.index(0) << std::endl;

		err->gradient(*output, target, outError);
		finalLayer()->backprop((MLStruct<double>*)&outError, 0.05);
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
		CPUMatrix* CPUInputsTemp = new CPUMatrix(GPUInputs->y, GPUInputs->x, GPUInputs->get_CPU_pointer());
		CPUMatrix* CPUOutputsTemp = new CPUMatrix(GPUOutputs->y, GPUOutputs->x, GPUOutputs->get_CPU_pointer());

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



void XORTest() {
	Matrix::forceUseCPU();
	Matrix ooi = Matrix(2, 1);
	ooi.setIndex(0, 1);
	ooi.setIndex(1, 1);
	Matrix ooo = Matrix(1, 1);
	ooo.setIndex(0, 0);

	Matrix ozi = Matrix(2, 1);
	ozi.setIndex(0, 1);
	ozi.setIndex(1, 0);
	Matrix ozo = Matrix(1, 1);
	ozo.setIndex(0, 1);

	Matrix zoi = Matrix(2, 1);
	zoi.setIndex(0, 0);
	zoi.setIndex(1, 1);
	Matrix zoo = Matrix(1, 1);
	zoo.setIndex(0, 1);

	Matrix zzi = Matrix(2, 1);
	zzi.setIndex(0, 0);
	zzi.setIndex(1, 0);
	Matrix zzo = Matrix(1, 1);
	zzo.setIndex(0, 0);

	Matrix outError = Matrix(1, 1);
	ErrorFunction* err = new ErrorHalfSquared;

	std::vector<size_t> layers = std::vector<size_t>();// = std::vector<size_t>({ 2,2,2,1 });
	layers.resize(3);
	layers[0] = 2;
	layers[1] = 2;
	layers[2] = 1;
	/*layers.resize(5);
	layers[0] = 2;
	layers[1] = 1024;
	layers[2] = 1024;
	layers[3] = 1024;
	layers[4] = 1;*/
	//layers[5] = 1;
	NeuralNetworkFF NN = NeuralNetworkFF(layers);
	
	NN.layers[0]->weights.setIndex(0, 0.45);
	NN.layers[0]->weights.setIndex(1, 0.2);
	NN.layers[0]->weights.setIndex(2, -0.6);
	NN.layers[0]->weights.setIndex(3, 0.8);
	NN.layers[0]->biases.setIndex(0, 0.2);
	NN.layers[0]->biases.setIndex(1, 0.7);
				
	NN.layers[1]->weights.setIndex(0, 0.2);
	NN.layers[1]->weights.setIndex(1, -0.8);
	NN.layers[1]->biases.setIndex(0, 0.3);

	
	int counter = 0;
	int iterations = 0;

	Timer::start();

	while (true) 
	{
		//Timer::start();
		NN.compute(ooi);
		//std::cout << "Compute: " << Timer::time();
		//NN.finalLayer().output.print();
		if (NN.finalLayer()->output.index(0) < 0.2) counter++;
		//Timer::start();
		NN.backprop(ooo, outError,err);
		//std::cout << "Backprop: " << Timer::time();

		NN.compute(ozi);
		////NN.finalLayer().output.print();
		if (NN.finalLayer()->output.index(0) > 0.8) counter++;
		NN.backprop(ozo, outError,err);

		NN.compute(zzi);
		////NN.finalLayer().output.print();
		if (NN.finalLayer()->output.index(0) < 0.2) counter++;
		NN.backprop(zzo, outError,err);
		

		NN.compute(zoi);
		////NN.finalLayer().output.print();
		if (NN.finalLayer()->output.index(0) > 0.8) counter++;
		NN.backprop(zoo, outError,err);
	
		

		if (counter == 4) {
			std::cout << "Time: " <<  Timer::time() << std::endl; //3085911 //8662048
			std::cout << "Finished: " << iterations << std::endl;
			//return;
		}
		else {
			iterations++;
			counter = 0;
		}

		std::cout << iterations << std::endl;

	}
}

void CNNTest() {
	Matrix::forceUseGPU();

	ConvLayer layer0(4,4,1,1,3,3,NULL);
	ConvLayer layer1(2,2,1,1,2,2,NULL);
	//NeuralNetwork* NN = new NeuralNetwork();
	NeuralNetwork NN;
	
	NN.addLayer(&layer0);
	NN.addLayer(&layer1);

	NN.randomise();

	Matrix in(4, 4);
	

	Matrix o(1, 1);
	Matrix t(1, 1);
	t.setIndex(0, 1);

	ErrorHalfSquared err;
	int i = 0;
	start:

	in.randomFill(0, 5);
	if (i % 2 == 0) {
		t.setIndex(0, -1);
	}
	else {
		t.setIndex(0, 1);
	}
	while (true)
	{
		i++;
		NN.compute(in);
		NN.backprop(t, o, &err);
		//std::cout << i << std::endl;
		if (abs(((Matrix*)NN.finalLayer()->getOutput())->index(0) - t.index(0)) < 0.01){
			std::cout << "Trained";
			goto start;
		}
		
	}
	

}



void BigCNNTest() {
	//Matrix::forceUseCPU();
	int d = 1000;
	ConvLayer layer0(d, d, 2, 1, d-1, d-1, NULL);
	ConvLayer layer1(2, 2, 1, 1, 2, 2, NULL);
	//NeuralNetwork* NN = new NeuralNetwork();
	NeuralNetwork NN;

	NN.addLayer(&layer0);
	NN.addLayer(&layer1);

	NN.randomise();

	Matrix in(d*2, d);


	Matrix o(1, 1);
	Matrix t(1, 1);
	t.setIndex(0, 1);

	ErrorHalfSquared err;
	int i = 0;
start:

	in.randomFill(0, 5);
	if (i % 2 == 0) {
		t.setIndex(0, -1);
	}
	else {
		t.setIndex(0, 1);
	}
	while (i < 10)
	{
		i++;
		NN.compute(in);
		NN.backprop(t, o, &err);
		std::cout << i << std::endl;
		//std::cout << i << std::endl;
		//if (abs(((Matrix*)NN.finalLayer()->getOutput())->index(0) - t.index(0)) < 0.03) {
		//	std::cout << "Trained";
		//	goto start;
		//}

	}


}