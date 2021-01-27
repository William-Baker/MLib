#pragma once
#include "Layer.hpp"

class ConvolutionalLayer : public ActiveLayer {
	
public:
	Matrix layer; //Convolution layer
	Matrix bias; //bias added to convolution output
	Matrix output; //the output of this layer
	Matrix layer_deltas; //error in the convoltuiional layer
	//Matrix unactivated_output; //output of the layer before activation function -- Not needed as not used
	Matrix gradient; //gradient at the output
	Matrix errorSignal;//The Error Signal at the input to this layer
	Matrix* input = 0;//Pointer to the last used input
	Matrix* error_signal_next_layer = 0;

	int inX;
	int inY;
	int inZ;

	int outX;
	int outY;
	int outZ;

	int convX;
	int convY;

	ConvolutionalLayer() : ActiveLayer(CV) {}

private:
	void allocate();
public:

	ConvolutionalLayer(int inX, int inY, int inZ, int outZ, int convX, int convY, Layer* prev_layer) : ConvolutionalLayer() {
		this->inX = inX;
		this->inY = inY;
		this->inZ = inZ;

		this->outX = inX - convX + 1;
		this->outY = inY - convY + 1;
		this->outZ = outZ;

		this->convX = convX;
		this->convY = convY;

		set_prev_layer(prev_layer);

		allocate();
		zeros();
	}

	ConvolutionalLayer(ConvolutionalLayer& m) = delete;




	virtual MLStruct<double>* get_error_signal() override {
		return errorSignal.getStrategy();
	}
	virtual MLStruct<double>* get_output() override {
		return output.getStrategy();
	}

	void randomise() override;

	void zeros() override;
	

	void compute(MLStruct<double>* inputIn) override;

	void compute() override;
	// void compute() override {
	// 	compute(get_prev_layer()->get_output());
	// }


	void backprop(Matrix &outError, double LR);
	
    void backprop(double LR) override;

	void backprop(MLStruct<double>* err, double LR) override;
	
	//Matrix layer;
	//Matrix bias;
	//Matrix output;
	//Matrix net;
	//Matrix errorSignal;//The Error Signal at teh input
	void print() override;

	void connect_to_next_layer(Layer* next);
	void connect_to_prev_layer(Layer* prev);

	~ConvolutionalLayer(){
		delete input;
	}


};