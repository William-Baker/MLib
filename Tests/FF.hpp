	
#pragma once
#include <thread>

#include "../Neural.hpp"
#include "Matrix Core.hpp"
    
namespace Test_FF{
    bool simpleXOR_CPU(){
		Matrix::forceUseCPU();
		XOR dataset;

		NeuralNetwork NN(new ErrorHalfSquared);
		NN.addLayer(new FeedForwardLayer(2,2,NULL));
		NN.addLayer(new FeedForwardLayer(2,1,NULL));

		NN.randomise();

		Trainer t(&dataset, &NN, 1000); //Data, model, batch size

		t.beginTraining(0.01, 0.5); //performance targe, LR
		return true;
	}

	bool simpleXOR_GPU(){
		Matrix::forceUseGPU();
		XOR dataset;

		NeuralNetwork NN(new ErrorHalfSquared);
		NN.addLayer(new FeedForwardLayer(2,2,NULL));
		NN.addLayer(new FeedForwardLayer(2,1,NULL));

		NN.randomise();

		Trainer t(&dataset, &NN, 1000); //Data, model, batch size

		t.beginTraining(0.01, 0.5); //performance targe, LR
		Matrix::forceUseCPU();
		return true;
	}
}