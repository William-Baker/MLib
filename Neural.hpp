#pragma once
#include "FeedForwardLayer.hpp"
#include "ConvolutionalLayer.hpp"


class ErrorHalfSquaredOutputLayer : public OutputLayer{
	Matrix error_differential;
	Matrix error;
	Matrix intermediate;
	public:
	ErrorHalfSquaredOutputLayer(size_t input_size) : OutputLayer(), error_differential(input_size, 1), error(input_size, 1), intermediate(input_size, 1) {}
	void computeErrorDifferential(MLStruct<double>* target) override{
		Matrix prev_out(get_prev_layer()->get_output(), false); //create temporary wrapper matrix for prev layer output matrix
		Matrix target_matrix(target, false); //create temporary wrapper matrix for target matrix
		target_matrix.subtract(prev_out, error_differential);
	}
	void computeError(MLStruct<double>* target) {
		Matrix prev_out(get_prev_layer()->get_output(), false); //create temporary wrapper matrix for prev layer output matrix
		Matrix target_matrix(target, false); //create temporary wrapper matrix for target matrix
		target_matrix.subtract(prev_out, error);
		error.multiply(error, intermediate);
		intermediate.scale(0.5, error);
	}
	MLStruct<double>* get_error_signal() override {return error_differential.getStrategy();}
	void print() override{
		std::cout << "--------------- Output Error ---------------" << std::endl;
		error.print();
	}

	MLStruct<double>* get_error() {return error.getStrategy(); }
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







//TODO refactor how layers are added
class NeuralNetwork {
	
public:
	std::vector<ActiveLayer*> layers;
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
		size_t output_layer_size = finalLayer()->get_output()->get_size();
		if((error_function == Default) || (error_function == ErrorHalfSquared)){
			output_layer = new ErrorHalfSquaredOutputLayer(output_layer_size);
		}
		output_layer->set_prev_layer(finalLayer());
	}
	
	void link_layers(Layer* left, Layer* right){
		if(dynamic_cast<FeedForwardLayer*>(right)){
			auto r = static_cast<FeedForwardLayer*>(right);
			r->connect_to_prev_layer(left);
		}
		else if (dynamic_cast<ConvolutionalLayer*>(right)){
			auto r = static_cast<ConvolutionalLayer*>(right);
			r->connect_to_prev_layer(left);			
		}

		if(dynamic_cast<FeedForwardLayer*>(left)){
			auto l = static_cast<FeedForwardLayer*>(left);
			l->connect_to_next_layer(right);
		}
		else if (dynamic_cast<ConvolutionalLayer*>(left)){
			auto l = static_cast<ConvolutionalLayer*>(left);
			l->connect_to_next_layer(right);
		}
	}

public:

	NeuralNetwork(std::vector<ActiveLayer*> layers, ErrorFunction err = Default) : NeuralNetwork() {
		error_function = err;
		this->layers = layers;
		Layer* prev_layer = &input_layer;
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->set_prev_layer(prev_layer);
			link_layers(prev_layer, layers[i]);
			prev_layer = layers[i];
		}
		generate_output_layer();
	}

	template<typename LayerType> void addLayer(LayerType* layer) {
		layers.push_back(layer);
		
		if (layers.size() == 1) layer->set_prev_layer(&input_layer);
		else layer->set_prev_layer(layers[layers.size() - 2]);

		link_layers(layers[layers.size() - 1]->get_prev_layer(), layers[layers.size() - 1]); //link this and prev
		
		delete output_layer; //delete the old output layer if applicable
		generate_output_layer();
		
		link_layers(layers[layers.size() - 1], output_layer); //link this and output
		
		std::cout << layer->get_output()->get_size();
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

		output_layer->computeErrorDifferential(target.getStrategy());

		finalLayer()->backprop(LR);

		return Matrix(output_layer->get_error_signal(), false).sum();

	}

	double get_error(Matrix& target) {
		output_layer->computeError(target.getStrategy());
		return Matrix(output_layer->get_error(), false).sum();
	}

	Layer* finalLayer() {
		return layers[layers.size() - 1];
	}

	Matrix get_output(){
		return Matrix(finalLayer()->get_output(), false);
	}

	void print() {
		input_layer.print();
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->print();
		}
		output_layer->print();
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


