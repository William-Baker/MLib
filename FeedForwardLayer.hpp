#pragma once
#include "Layer.hpp"

class FeedForwardLayer : public ActiveLayer {
public:
	Matrix weights;
	Matrix biases;
	Matrix output;
	Matrix net;
	Matrix error_signal; //The error signal at the input
	Matrix temp2;
	Matrix gradient;
	Matrix weights_deltas;
	Matrix* input = 0;
	Matrix* error_signal_next_layer = 0;
	
	
	/* FeedForwardLayer& operator=(FeedForwardLayer&& m){
		std::swap(this->weights, m.weights);
		std::swap(this->biases, m.biases);
		std::swap(this->output, m.output);
		std::swap(this->net , m.net);
		std::swap(this->error_signal , m.error_signal);
		std::swap(this->temp2 , m.temp2);
		std::swap(this->gradient , m.gradient);
		std::swap(this->weights_deltas , m.weights_deltas);
		std::swap(this->next_layer , m.next_layer);
		std::swap(this->prev_layer , m.prev_layer);
		std::swap(this->type , m.type);

		return *this;
	} */
	FeedForwardLayer& operator=(FeedForwardLayer&& m) = default;

	FeedForwardLayer() : ActiveLayer(FF){}
	FeedForwardLayer(size_t inputs, size_t outputs, Layer* prev_layer) : FeedForwardLayer() {
		weights = Matrix(outputs, inputs);
		weights_deltas = Matrix(outputs, inputs);
		biases = Matrix(outputs, 1);
		output = Matrix(outputs, 1);
		error_signal = Matrix(inputs, 1);
		net = Matrix(outputs, 1);
		temp2 = Matrix(outputs, 1);
		gradient = Matrix(outputs, 1);
		zeros();
		set_prev_layer(prev_layer);
	}

	FeedForwardLayer(Matrix& Weights, Matrix& Biases, Layer* prev_layer) : FeedForwardLayer(){
		weights = std::move(Weights);
		weights_deltas = Matrix(weights.height(), weights.width());
		biases = Matrix(weights.height(), 1);
		output = Matrix(weights.height(), 1);
		error_signal = Matrix(weights.width(), 1);
		net = Matrix(weights.height(), 1);
		temp2 = Matrix(weights.height(), 1);
		gradient = Matrix(weights.height(), 1);
		zeros();
		set_prev_layer(prev_layer);
	}







	virtual MLStruct<double>* get_error_signal() override {
		return error_signal.getStrategy();
	}
	virtual MLStruct<double>* get_output() override {
		return output.getStrategy();
	}
	





	void randomise() override;

	void zeros() override;
	
	// virtual void compute() override {
	// 	Matrix tmp(get_prev_layer()->get_output(), false);
	// 	compute(tmp);
	// }

	void compute(MLStruct<double>* inputIn) override;
	void compute() override;

	void calculateErrorSignal(Matrix& outputError);

	void update(double LR, Matrix& errorSigAtOutput);
	void backprop(double LR) override;
	/**
	 * @deprecated
	 */
	void backprop(MLStruct<double>* err, double LR) override;
	/**
	 * @deprecated
	 */
	void backprop(Matrix& outErrorSig, double LR);
	
	

	void print() override;


	void connect_to_next_layer(Layer* next);

	void connect_to_prev_layer(Layer* prev);

	
};