#pragma once
#include "Training.hpp"
#include <thread>
#include "include/IO.hpp"
#include "CPUTensor.hpp"
#include "GPUTensor.hpp"
	

namespace Test {

	
	void conv() {
		std::cout << "----- Conv ----" << std::endl;
		{
			Matrix::forceUseCPU();

			ConvLayer cv(3, 3, 1, 1, 2, 2,0);
			cv.layer.setIndex(0, 0.2);
			cv.layer.setIndex(1, 0.4);
			cv.layer.setIndex(2, 0.3);
			cv.layer.setIndex(3, 0.5);

			cv.bias.setIndex(0, -0.2);

			Matrix input(3, 3);
			input.setIndex(0, 0.3);
			input.setIndex(1, 0.2);
			input.setIndex(2, -0.5);
			input.setIndex(3, 0.3);
			input.setIndex(4, 0.9);
			input.setIndex(5, 0.1);
			input.setIndex(6, 0.4);
			input.setIndex(7, 0);
			input.setIndex(8, -0.2);

			Matrix output(2, 2);
			output.setIndex(0, tanh(0.48));
			output.setIndex(1, tanh(-0.04));
			output.setIndex(2, tanh(0.34));
			output.setIndex(3, tanh(-0.08));

			cv.compute(input);

			std::cout << "----- Error report ----" << std::endl;

			cv.output.print();

			std::cout << Matrix::compare(output, cv.output, -0.09, 0.5);

			std::cout << "-----------------------" << std::endl;
		}

		{
			Matrix::forceUseGPU();

			ConvLayer cv(3, 3, 1, 1, 2, 2,0);
			cv.layer.setIndex(0, 0.2);
			cv.layer.setIndex(1, 0.4);
			cv.layer.setIndex(2, 0.3);
			cv.layer.setIndex(3, 0.5);

			cv.bias.setIndex(0, -0.2);

			Matrix input(3, 3);
			input.setIndex(0, 0.3);
			input.setIndex(1, 0.2);
			input.setIndex(2, -0.5);
			input.setIndex(3, 0.3);
			input.setIndex(4, 0.9);
			input.setIndex(5, 0.1);
			input.setIndex(6, 0.4);
			input.setIndex(7, 0);
			input.setIndex(8, -0.2);

			Matrix output(2, 2);
			output.setIndex(0, tanh(0.48));
			output.setIndex(1, tanh(-0.04));
			output.setIndex(2, tanh(0.34));
			output.setIndex(3, tanh(-0.08));

			cv.compute(input);

			std::cout << "----- Error report ----" << std::endl;

			cv.output.print();

			std::cout << Matrix::compare(output, cv.output, -0.09, 0.5);

			std::cout << "-----------------------" << std::endl;
		}

		{
			Matrix::forceUseCPU();

			ConvLayer cv(3, 3, 2, 1, 2, 2,0);
			cv.layer.setIndex(0, 0.2);
			cv.layer.setIndex(1, 0.2);
			cv.layer.setIndex(2, 0.4);
			cv.layer.setIndex(3, 0.4);
			cv.layer.setIndex(4, 0.3);
			cv.layer.setIndex(5, 0.3);
			cv.layer.setIndex(6, 0.5);
			cv.layer.setIndex(7, 0.5);

			cv.bias.setIndex(0, 0);

			Matrix input(6, 3);
			input.setIndex(0, 0.3);
			input.setIndex(1, 0.3);
			input.setIndex(2, 0.2);
			input.setIndex(3, 0.2);
			input.setIndex(4, -0.5);
			input.setIndex(5, -0.5);
			input.setIndex(6, 0.3);
			input.setIndex(7, 0.3);
			input.setIndex(8, 0.9);
			input.setIndex(9, 0.9);
			input.setIndex(10, 0.1);
			input.setIndex(11, 0.1);
			input.setIndex(12, 0.4);
			input.setIndex(13, 0.4);
			input.setIndex(14, 0);
			input.setIndex(15, 0);
			input.setIndex(16, -0.2);
			input.setIndex(17, -0.2);
			

			Matrix output(2, 2);
			output.setIndex(0, tanh(1.36)); //0.38  0.2*0.3+0.4*0.2+0.3*-0.5+0.5*0.3+-0.2*0.4+0+0.1*-0.2-0.4 = -0.36
			output.setIndex(1, tanh(0.32));
			output.setIndex(2, tanh(1.08));
			output.setIndex(3, tanh(0.24));

			cv.compute(input);

			std::cout << "----- Error report ----" << std::endl;

			cv.output.print();

			std::cout << Matrix::compare(output, cv.output, -1, 1);

			std::cout << "-----------------------" << std::endl;
		}

		{
			Matrix::forceUseGPU();

			ConvLayer cv(3, 3, 2, 1, 2, 2,0);
			cv.layer.setIndex(0, 0.2);
			cv.layer.setIndex(1, 0.2);
			cv.layer.setIndex(2, 0.4);
			cv.layer.setIndex(3, 0.4);
			cv.layer.setIndex(4, 0.3);
			cv.layer.setIndex(5, 0.3);
			cv.layer.setIndex(6, 0.5);
			cv.layer.setIndex(7, 0.5);

			cv.bias.setIndex(0, 0);

			Matrix input(6, 3);
			input.setIndex(0, 0.3);
			input.setIndex(1, 0.3);
			input.setIndex(2, 0.2);
			input.setIndex(3, 0.2);
			input.setIndex(4, -0.5);
			input.setIndex(5, -0.5);
			input.setIndex(6, 0.3);
			input.setIndex(7, 0.3);
			input.setIndex(8, 0.9);
			input.setIndex(9, 0.9);
			input.setIndex(10, 0.1);
			input.setIndex(11, 0.1);
			input.setIndex(12, 0.4);
			input.setIndex(13, 0.4);
			input.setIndex(14, 0);
			input.setIndex(15, 0);
			input.setIndex(16, -0.2);
			input.setIndex(17, -0.2);


			Matrix output(2, 2);
			output.setIndex(0, tanh(1.36)); //0.38  0.2*0.3+0.4*0.2+0.3*-0.5+0.5*0.3+-0.2*0.4+0+0.1*-0.2-0.4 = -0.36
			output.setIndex(1, tanh(0.32));
			output.setIndex(2, tanh(1.08));
			output.setIndex(3, tanh(0.24));

			cv.compute(input);

			std::cout << "----- Error report ----" << std::endl;

			cv.output.print();

			std::cout << Matrix::compare(output, cv.output, -1, 1);

			std::cout << "-----------------------" << std::endl;
		}


		{
			Matrix::forceUseCPU();

			ConvLayer cv(3, 3, 2, 2, 2, 2,0);
			cv.layer.setIndex(0, 0.2);
			cv.layer.setIndex(1, 0.2);
			cv.layer.setIndex(2, 0.4);
			cv.layer.setIndex(3, 0.4);
			cv.layer.setIndex(4, 0.3);
			cv.layer.setIndex(5, 0.3);
			cv.layer.setIndex(6, 0.5);
			cv.layer.setIndex(7, 0.5);
			cv.layer.setIndex(8, 0.2);
			cv.layer.setIndex(9, 0.2);
			cv.layer.setIndex(10, 0.4);
			cv.layer.setIndex(11, 0.4);
			cv.layer.setIndex(12, 0.3);
			cv.layer.setIndex(13, 0.3);
			cv.layer.setIndex(14, 0.5);
			cv.layer.setIndex(15, 0.5);

			cv.bias.setIndex(0, 0);
			cv.bias.setIndex(1, 0);

			Matrix input(6, 3);
			input.setIndex(0, 0.3);
			input.setIndex(1, 0.3);
			input.setIndex(2, 0.2);
			input.setIndex(3, 0.2);
			input.setIndex(4, -0.5);
			input.setIndex(5, -0.5);
			input.setIndex(6, 0.3);
			input.setIndex(7, 0.3);
			input.setIndex(8, 0.9);
			input.setIndex(9, 0.9);
			input.setIndex(10, 0.1);
			input.setIndex(11, 0.1);
			input.setIndex(12, 0.4);
			input.setIndex(13, 0.4);
			input.setIndex(14, 0);
			input.setIndex(15, 0);
			input.setIndex(16, -0.2);
			input.setIndex(17, -0.2);


			Matrix output(4, 2);
			output.setIndex(0, tanh(1.36)); 
			output.setIndex(1, tanh(1.36));
			output.setIndex(2, tanh(0.32));
			output.setIndex(3, tanh(0.32));
			output.setIndex(4, tanh(1.08));
			output.setIndex(5, tanh(1.08));
			output.setIndex(6, tanh(0.24));
			output.setIndex(7, tanh(0.24));
		

			cv.compute(input);

			std::cout << "----- Error report ----" << std::endl;

			cv.output.print();

			std::cout << Matrix::compare(output, cv.output, -1, 1);

			std::cout << "-----------------------" << std::endl;
		}


		{
			Matrix::forceUseGPU();

			ConvLayer cv(3, 3, 2, 2, 2, 2,NULL);
			cv.layer.setIndex(0, 0.2);
			cv.layer.setIndex(1, 0.2);
			cv.layer.setIndex(2, 0.4);
			cv.layer.setIndex(3, 0.4);
			cv.layer.setIndex(4, 0.3);
			cv.layer.setIndex(5, 0.3);
			cv.layer.setIndex(6, 0.5);
			cv.layer.setIndex(7, 0.5);
			cv.layer.setIndex(8, 0.2);
			cv.layer.setIndex(9, 0.2);
			cv.layer.setIndex(10, 0.4);
			cv.layer.setIndex(11, 0.4);
			cv.layer.setIndex(12, 0.3);
			cv.layer.setIndex(13, 0.3);
			cv.layer.setIndex(14, 0.5);
			cv.layer.setIndex(15, 0.5);

			cv.bias.setIndex(0, 0);
			cv.bias.setIndex(1, 0);

			Matrix input(6, 3);
			input.setIndex(0, 0.3);
			input.setIndex(1, 0.3);
			input.setIndex(2, 0.2);
			input.setIndex(3, 0.2);
			input.setIndex(4, -0.5);
			input.setIndex(5, -0.5);
			input.setIndex(6, 0.3);
			input.setIndex(7, 0.3);
			input.setIndex(8, 0.9);
			input.setIndex(9, 0.9);
			input.setIndex(10, 0.1);
			input.setIndex(11, 0.1);
			input.setIndex(12, 0.4);
			input.setIndex(13, 0.4);
			input.setIndex(14, 0);
			input.setIndex(15, 0);
			input.setIndex(16, -0.2);
			input.setIndex(17, -0.2);


			Matrix output(4, 2);
			output.setIndex(0, tanh(1.36));
			output.setIndex(1, tanh(1.36));
			output.setIndex(2, tanh(0.32));
			output.setIndex(3, tanh(0.32));
			output.setIndex(4, tanh(1.08));
			output.setIndex(5, tanh(1.08));
			output.setIndex(6, tanh(0.24));
			output.setIndex(7, tanh(0.24));


			cv.compute(input);

			std::cout << "----- Error report ----" << std::endl;

			cv.output.print();

			std::cout << Matrix::compare(output, cv.output, -1, 1);

			std::cout << "-----------------------" << std::endl;
		}
		

	}

	void convBackprop(){
		//as soon as error starts at -1 or more - fucked even -0.95
		std::cout << "----- Convolutional Backprop -----" << std::endl;
		Matrix::forceUseCPU();
		//nt inX, int inY, int inZ, int outZ, int convX, int convY, Layer* prevLayer
		ConvLayer cv(3,3,2,1,2,2,NULL);
		cv.randomise();
		Matrix input(6,3);
		input.randomFill(0,1);
		std::cout << "Inputs" << std::endl;
		input.print();
		std::cout << std::endl << std::endl;
		Matrix target(2, 2);
		target.setIndex(0, 1);
		target.setIndex(1, 1);
		target.setIndex(2, -1);
		target.setIndex(3, 1);
	

		double err = 10;

		Matrix error(2,2);
		


		int counter = 0;

		while(err > 0.2){
			cv.compute(input);
			target.subtract(cv.output, error);
			err = abs(error.index(0)) + abs(error.index(1)) + abs(error.index(2)) + abs(error.index(3));
			cv.backprop(error, 0.1);
			counter ++;
			std::cout << "\nIteration: " << counter;
			std::cout << " Error:\n";
			error.print();

		}
		
		std::cout << "-----------------------" << std::endl;
	}

	void tensor_copy(){
		std::cout << "----- Tensor copy -----" << std::endl;
		Matrix::forceUseCPU();
		CPUTensor a(3,3,2);
		a.randomFill(0,10);
		CPUTensor b = CPUTensor::copy(a);

		GPUTensor ga(3,3,2);
		ga.randomFill(0,10);
		GPUTensor gb = GPUTensor::copy(ga);

		std::cout << "----- CPU Error report ----" << std::endl;
		Matrix::forceUseGPU();
		Matrix ma(a.get_implementation());
		Matrix mb(b.get_implementation());
		std::cout << Matrix::compare(ma, mb, -1, 1);

		std::cout << "----- GPU Error report ----" << std::endl;

		Matrix mga(ga.get_implementation());
		Matrix mgb(gb.get_implementation());
		std::cout << Matrix::compare(mga, mgb, -1, 1);

		Matrix::forceUseCPU();

		ConvLayer cv(3, 3, 2, 2, 2, 2,NULL);
		cv.layer.setIndex(0, 0.2);
		cv.layer.setIndex(1, 0.2);
		cv.layer.setIndex(2, 0.4);
		cv.layer.setIndex(3, 0.4);
		cv.layer.setIndex(4, 0.3);
		cv.layer.setIndex(5, 0.3);
		cv.layer.setIndex(6, 0.5);
		cv.layer.setIndex(7, 0.5);
		cv.layer.setIndex(8, 0.2);
		cv.layer.setIndex(9, 0.2);
		cv.layer.setIndex(10, 0.4);
		cv.layer.setIndex(11, 0.4);
		cv.layer.setIndex(12, 0.3);
		cv.layer.setIndex(13, 0.3);
		cv.layer.setIndex(14, 0.5);
		cv.layer.setIndex(15, 0.5);

		cv.bias.setIndex(0, 0);
		cv.bias.setIndex(1, 0);

		Matrix input(6, 3);
		input.setIndex(0, 0.3);
		input.setIndex(1, 0.3);
		input.setIndex(2, 0.2);
		input.setIndex(3, 0.2);
		input.setIndex(4, -0.5);
		input.setIndex(5, -0.5);
		input.setIndex(6, 0.3);
		input.setIndex(7, 0.3);
		input.setIndex(8, 0.9);
		input.setIndex(9, 0.9);
		input.setIndex(10, 0.1);
		input.setIndex(11, 0.1);
		input.setIndex(12, 0.4);
		input.setIndex(13, 0.4);
		input.setIndex(14, 0);
		input.setIndex(15, 0);
		input.setIndex(16, -0.2);
		input.setIndex(17, -0.2);

		cv.compute(input);

		CPUTensor in(static_cast<CPUMatrix*>(input.getStrategy()), 2);
		CPUTensor layer(static_cast<CPUMatrix*>(cv.layer.getStrategy()), 2, 2);
		CPUTensor bias(static_cast<CPUMatrix*>(cv.bias.getStrategy()), 1);
		CPUTensor ot(2,2,2);
		in.convolute(&layer, &bias, &ot);


		cv.output.print();
		ot.print();

		

		std::cout << "-----------------------" << std::endl;
	}

//TODO recheck for GPU
	void testAll() {
		conv();
		convBackprop();
	}



	void simpleXOR(){
		XOR dataset;

		NeuralNetwork NN(new ErrorHalfSquared);
		NN.addLayer(new FeedForwardLayer(2,2,NULL));
		NN.addLayer(new FeedForwardLayer(2,1,NULL));

		NN.randomise();

		Trainer t(&dataset, &NN, 1000);
		int c = 0;

		t.beginTraining(0.01, 0.5);
		std::cout << "trained: " << c++ << std::endl;
		NN.randomise();

	}

	void XORTest() {
		std::cout << "----- XOR Backprop -----" << std::endl;
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

		XOR d;
		while (true) 
		{
			//Timer::start();
			// NN.compute(ooi.getStrategy());
			// //std::cout << "Compute: " << Timer::time();
			// //NN.finalLayer().output.print();
			// if (NN.finalLayer()->output.index(0) < 0.01) counter++;
			// //Timer::start();
			// NN.backprop(ooo, outError,err);
			// //std::cout << "Backprop: " << Timer::time();

			// NN.compute(ozi.getStrategy());
			// ////NN.finalLayer().output.print();
			// if (NN.finalLayer()->output.index(0) > 0.99) counter++;
			// NN.backprop(ozo, outError,err);

			// NN.compute(zzi.getStrategy());
			// ////NN.finalLayer().output.print();
			// if (NN.finalLayer()->output.index(0) < 0.01) counter++;
			// NN.backprop(zzo, outError,err);
			

			// NN.compute(zoi.getStrategy());
			// ////NN.finalLayer().output.print();
			// if (NN.finalLayer()->output.index(0) > 0.99) counter++;
			// NN.backprop(zoo, outError,err);
			auto cas = d.get_next_case();
			NN.compute(cas.inputs);
			//std::cout << "Compute: " << Timer::time();
			//NN.finalLayer().output.print();
			if (NN.finalLayer()->output.index(0) > 0.99) counter++;
			//Timer::start();
			Matrix tmp(cas.outputs, false);
			NN.backprop(tmp, outError,err);
			//std::cout << "Backprop: " << Timer::time();

			cas = d.get_next_case();
			NN.compute(cas.inputs);
			////NN.finalLayer().output.print();
			tmp = Matrix(cas.outputs, false);
			if (NN.finalLayer()->output.index(0) > 0.99) counter++;
			NN.backprop(tmp, outError,err);

			cas = d.get_next_case();
			NN.compute(cas.inputs);
			////NN.finalLayer().output.print();
			tmp = Matrix(cas.outputs, false);
			if (NN.finalLayer()->output.index(0) < 0.01) counter++;
			NN.backprop(tmp, outError,err);
			
			cas = d.get_next_case();
			NN.compute(cas.inputs);
			////NN.finalLayer().output.print();
			tmp = Matrix(cas.outputs, false);
			if (NN.finalLayer()->output.index(0) < 0.01) counter++;
			NN.backprop(tmp, outError,err);
			

			if (counter == 4) {
				std::cout << "Time: " <<  Timer::time() << std::endl; //3085911 //8662048
				std::cout << "Finished: " << iterations << std::endl;
				NN.randomise();
				counter = 0;
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

		ErrorHalfSquared err;

		NeuralNetwork NN(&err);
		
		NN.addLayer(&layer0);
		NN.addLayer(&layer1);

		NN.randomise();

		Matrix in(4, 4);
		

		Matrix o(1, 1);
		Matrix t(1, 1);
		t.setIndex(0, 1);

		
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
			NN.compute(in.getStrategy());
			NN.backprop(t);
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
		ErrorHalfSquared err;
		NeuralNetwork NN(&err);

		NN.addLayer(&layer0);
		NN.addLayer(&layer1);

		NN.randomise();

		Matrix in(d*2, d);


		Matrix o(1, 1);
		Matrix t(1, 1);
		t.setIndex(0, 1);

		
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
			NN.compute(in.getStrategy());
			NN.backprop(t);
			std::cout << i << std::endl;
			//std::cout << i << std::endl;
			//if (abs(((Matrix*)NN.finalLayer()->getOutput())->index(0) - t.index(0)) < 0.03) {
			//	std::cout << "Trained";
			//	goto start;
			//}

		}


	}
}