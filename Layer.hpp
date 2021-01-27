#pragma once
#include "Matrix.hpp"


class Layer {
	Layer* next_layer;
	Layer* prev_layer;
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
		next_layer = 0;
		prev_layer = 0;
	}

	virtual void compute() = 0;
	virtual void compute(MLStruct<double>* in) = 0;
	virtual void backprop(double LR) = 0;
	virtual void backprop(MLStruct<double>* err, double LR) = 0;
	
	// virtual MLStruct<double>* getInputError() = 0; //instead we require a pointer to the next layers error signal
	virtual MLStruct<double>* get_error_signal() = 0;
	virtual MLStruct<double>* get_output() = 0;
	virtual void print() = 0;


	inline void set_next_layer(Layer* layer) {
		next_layer = layer;
		if(layer) layer->prev_layer = this;
	}
	inline void set_prev_layer(Layer* layer) {
		prev_layer = layer;
		if(layer) layer->next_layer = this;
	}

	

	inline Layer* get_next_layer() {return next_layer;}
	inline Layer* get_prev_layer() {return prev_layer;}
};

/**
 * represents an ML layer that does processing
 */
class ActiveLayer : public Layer {
public:
	ActiveLayer(LayerType typeIn) : Layer(typeIn){}
	virtual void zeros() = 0;
	virtual void randomise() = 0;
};

class InputLayer : public Layer {
public:
	MLStruct<double>* output = 0;

	// InputLayer& operator=(InputLayer& m) {
	// 	std::swap(this->output, m.output);
	// 	std::swap(this->next_layer, m.next_layer);
	// 	std::swap(this->prev_layer, m.prev_layer);
	// 	std::swap(this->type, m.type);
	// 	return *this;
	// }

	InputLayer() : Layer(INPUT){
	}

	InputLayer(Matrix* output) : InputLayer() {
		this->output = output->getStrategy();
	}

	InputLayer(MLStruct<double>* output) : InputLayer() {
		this->output = output;
	}

	/**
	 * Just passes the computation forward to the next element
	 */
	void compute() override {
		if (get_next_layer()) {
			get_next_layer()->compute();
		}
	}

	/**
	 * Just passes the computation forward to the next element, without modifying the input
	 */
	void compute(MLStruct<double>* input) override {
		output = input; //Set the output as the input directly as this layer does nothing
		if (get_next_layer()) {
			get_next_layer()->compute(input);
		}
	}

	void backprop(MLStruct<double>* err, double LR) override {
		return;
	}

	void backprop(double LR) override {
		return;
	}

	MLStruct<double>* get_error_signal() override {
		return nullptr;
	}

	MLStruct<double>* get_output() override {
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

	virtual void computeErrorDifferential(MLStruct<double>* target) = 0;
	virtual void computeError(MLStruct<double>* target) = 0;

	/**
	 * Pass backprop to previous element without modification
	 */
	void backprop(MLStruct<double>* err, double LR) override {
		get_prev_layer()->backprop(err, LR);
	}

	/**
	 * Pass backprop to previous element without modification
	 */
	void backprop(double LR) override {
		get_prev_layer()->backprop(LR);
	}


	MLStruct<double>* get_output() override {
		return get_prev_layer()->get_output();
	}

    virtual MLStruct<double>* get_error() = 0;

};