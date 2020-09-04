#pragma once
#include "Neural.hpp"
#include <thread>
#include <IO.hpp>

	

namespace Test {

	void trans() {
		std::cout << "---------- Trans ----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(4, 3);
		CA.randomFill(-10, 10);
#ifdef DEBUG
		CA.print();
		std::cout << std::endl;
#endif
		Matrix CC = CA.transposeNew();
#ifdef DEBUG
		CC.print();
		std::cout << std::endl;
		Matrix::forceUseGPU();
#endif
		Matrix A;
		A.copyToThis(CA);
		Matrix C = A.transposeNew();
#ifdef DEBUG
		C.print();
#endif
		std::cout << "Comparison: " << Matrix::compare(CC, C, 0, 1) << std::endl << std::endl;
	}

	void transMult() {
		std::cout << "---------- Trans Mult ----------" << std::endl;
		Matrix A(3, 5);
		A.randomFill(0, 10);
		Matrix B(5, 4);
		B.randomFill(0, 10);
#ifdef DEBUG
		A.print();
		std::cout << "\n\n";
		B.print();
		std::cout << "\n\n";
#endif
		auto A_t = A.transposeNew();
#ifdef DEBUG
		A_t.print();
		std::cout << "\n\n";
#endif
		auto B_t = B.transposeNew();
#ifdef DEBUG
		B_t.print();
		std::cout << "\n\n";
#endif

		auto normal = A.multiply(B);
		auto amult = A_t.multiplyA(B);
		auto bmult = A.multiplyB(B_t);
		auto abmult = A_t.multiplyAB(B_t);

#ifdef DEBUG
		normal.print();
		std::cout << "\n\n";
		amult.print();
		std::cout << "\n\n";
		bmult.print();
		std::cout << "\n\n";
		abmult.print();
		std::cout << "\n\n";
#endif
		std::cout << "Compare: " << Matrix::compare(normal, amult, 0, 300) << std::endl;
		std::cout << "Compare: " << Matrix::compare(normal, bmult, 0, 300) << std::endl;
		std::cout << "Compare: " << Matrix::compare(normal, abmult, 0, 300) << std::endl;
	}

	void mult() {
		std::cout << "---------- Mult ----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(2, 3);
		CA.randomFill(0, 10);
		CA.print(20);
		Matrix CB(3, 2);
		CB.randomFill(0, 10);
		CB.print(20);

		Matrix C = CA.multiply(CB);
		
		C.print(20);
		Matrix::forceUseGPU();

		Matrix A, B;
		A.copyToThis(CA);
		B.copyToThis(CB);
		Matrix arr(A.height(), B.width());
		A.multiply(B, arr);
		arr.print(20);
		std::cout << Matrix::compare(arr, C, 0, 9*10);
		arr.~Matrix();
		
	}

	void multMany() {
		std::cout << "---------- Mult Many Big ----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(1000, 500);
		CA.randomFill(0, 10);

		Matrix CB(500, 450);
		CB.randomFill(0, 10);

		std::cout << "CPU Starting: ";
		Timer::start();
		Matrix C = CA.multiply(CB);
		size_t CPU_time = Timer::time();
		std::cout << "CPU Time: " << CPU_time << "us" << std::endl;
		
		Matrix::forceUseGPU();
		Matrix A, B;
		A.copyToThis(CA);
		B.copyToThis(CB);
		const int count = 100;
		Matrix* arr[count];
		
		std::cout << "allocating matrices: ";
		for (int i = 0; i < count; i++) {
			if(!(i%(count/10))) std::cout << i << ",";
			arr[i] = new Matrix(A.height(), B.width());
		}
		std::cout << std::endl;

		A.multiply(B, *arr[0]);//Warm up
		Timer::start();
		fprintf(stderr, "GPU multiplying: ");
		for (int i = 0; i < count; i++) {
			if(!(i%(count/10))) fprintf(stderr, "%i,", i);
			A.multiply(B, *arr[i]);
			cudaStreamSynchronize(GPUMatrix::stream);
		}
		fprintf(stderr, "\n");
		size_t GPU_average_time = Timer::time()/count;
		std::cout << "Average time: " << GPU_average_time << "us" << std::endl;
		std::cout << "GPU / CPU: " << CPU_time / GPU_average_time << std::endl;
		#ifdef DEBUG
		std::cout << "----- Error report ----" << std::endl;
		for (int i = 0; i < count; i++) {
			std::cout << Matrix::compare(*arr[i], C, 0, 500*500*10*10);
		}
		std::cout << "-----------------------" << std::endl;
		#endif

	    std::cout << "de-allocating matrices: ";
		for (int i = 0; i < count; i++) {
			if(!(i%(count/10))) std::cout << i;
			arr[i]->~Matrix();
		}
		std::cout << std::endl;
	}

	void multManySmall() {
		std::cout << "---------- Mult Many Small----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(10, 50);
		CA.randomFill(0, 10);

		Matrix CB(50, 45);
		CB.randomFill(0, 10);

		std::cout << "CPU Starting: ";
		Timer::start();
		Matrix C = CA.multiply(CB);
		size_t CPU_time = Timer::time();
		std::cout << "CPU Time: " << CPU_time << "us" << std::endl;
		
		Matrix::forceUseGPU();
		Matrix A, B;
		A.copyToThis(CA);
		B.copyToThis(CB);
		const int count = 100;
		Matrix* arr[count];
		
		std::cout << "allocating matrices: ";
		for (int i = 0; i < count; i++) {
			if(!(i%(count/10))) std::cout << i << ",";
			arr[i] = new Matrix(A.height(), B.width());
		}
		std::cout << std::endl;

		A.multiply(B, *arr[0]);//Warm up
		Timer::start();
		fprintf(stderr, "GPU multiplying: ");
		for (int i = 0; i < count; i++) {
			if(!(i%(count/10))) fprintf(stderr, "%i,", i);
			A.multiply(B, *arr[i]);
			cudaStreamSynchronize(GPUMatrix::stream);
		}
		fprintf(stderr, "\n");
		size_t GPU_average_time = Timer::time()/count;
		std::cout << "Average time: " << GPU_average_time << "us" << std::endl;
		std::cout << "GPU / CPU: " << CPU_time / GPU_average_time << std::endl;
		std::cout << "----- Error report ----" << std::endl;
		#ifdef DEBUG
		for (int i = 0; i < count; i++) {
			std::cout << Matrix::compare(*arr[i], C, 0, 50*50*10*10);
		}
		#else
		for (int i = count-1; i < count; i++) {
			std::cout << Matrix::compare(*arr[i], C, 0, 50*50*10*10);
		}
		#endif
			
		std::cout << "-----------------------" << std::endl;
		
	    std::cout << "de-allocating matrices: ";
		for (int i = 0; i < count; i++) {
			if(!(i%(count/10))) std::cout << i;
			arr[i]->~Matrix();
		}
		std::cout << std::endl;
	}

	void multElementMany(){
		std::cout << "---------- Element Many Big ----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(1000, 500);
		CA.randomFill(0, 10);

		Matrix CB(1000, 500);
		CB.randomFill(0, 10);

		std::cout << "CPU Starting: ";
		Timer::start();
		Matrix C = CA.multiplyElementWise(CB);
		size_t CPU_time = Timer::time();
		std::cout << "CPU Time: " << CPU_time << "us" << std::endl;
		
		Matrix::forceUseGPU();
		Matrix A, B;
		A.copyToThis(CA);
		B.copyToThis(CB);
		const int count = 100;
		Matrix* arr[count];
		
		std::cout << "allocating matrices: ";
		for (int i = 0; i < count; i++) {
			if(!(i%(count/10))) std::cout << i << ",";
			arr[i] = new Matrix(A.height(), B.width());
		}
		std::cout << std::endl;

		A.multiplyElementWise(B, *arr[0]);//Warm up
		Timer::start();
		fprintf(stderr, "GPU multiplying: ");
		for (int i = 0; i < count; i++) {
			if(!(i%(count/10))) fprintf(stderr, "%i,", i);
			A.multiplyElementWise(B, *arr[i]);
			cudaStreamSynchronize(GPUMatrix::stream);
		}
		fprintf(stderr, "\n");
		size_t GPU_average_time = Timer::time()/count;
		std::cout << "Average time: " << GPU_average_time << "us" << std::endl;
		std::cout << "GPU / CPU: " << CPU_time / GPU_average_time << std::endl;

		std::cout << "----- Error report ----" << std::endl;
		#ifdef DEBUG
		for (int i = 0; i < count; i++) {
			std::cout << Matrix::compare(*arr[i], C, 0, 10*10);
		}
		#else
		for (int i = count-1; i < count; i++) {
			std::cout << Matrix::compare(*arr[i], C, 0, 10*10);
		}
		#endif
			
		std::cout << "-----------------------" << std::endl;
		

	    std::cout << "de-allocating matrices: ";
		for (int i = 0; i < count; i++) {
			if(!(i%(count/10))) std::cout << i;
			arr[i]->~Matrix();
		}
		std::cout << std::endl;
	}

/* 
	void ElementTimed() {
		std::cout << "---------- Mult ----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(10000, 2000);
		CA.randomFill(0, 1000000);
		Matrix CB(10000, 2000);
		CB.randomFill(0, 1000000);
		std::cout << "CPU Starting: ";
		Timer::start();
		Matrix C = CA.multiplyElementWise(CB);
		std::cout << "CPU Time: " << Timer::time() << std::endl;

		Matrix::forceUseGPU();
		Matrix A, B;
		A.copyToThis(CA);
		B.copyToThis(CB);
		const int n = 10;
		Matrix* arr[n];
		for (int i = 0; i < n; i++) {
			//std::cout << "allocating matrix: " << i << std::endl;
			arr[i] = new Matrix(A.height(), B.width());
		}
		A.multiplyElementWise(B, *arr[0]);
		Timer::start();
		for (int i = 0; i < n; i++) {
			//std::cout << "GPU multiplying: " << i << std::endl;
			A.multiplyElementWise(B, *arr[i]);

			//std::cout << Matrix::compare(normal, CM, 0, 5000) << std::endl;
		}
		std::cout << Timer::time() / n << std::endl;
		std::cout << "----- Error report ----" << std::endl;
		for (int i = 0; i < n; i++) {
			std::cout << Matrix::compare(*arr[i], C, 0, 5000);
		}
		std::cout << "-----------------------" << std::endl;

		for (int i = 0; i < n; i++) {
			arr[i]->~Matrix();
		}
	} */
	void multElementWise() {
		std::cout << "-------- Element Wise --------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(4, 3);
		CA.randomFill(-10, 10);
		Matrix CB(4, 3);
		CB.randomFill(-10, 10);
		Matrix CC(CA.height(), CA.width());
		CA.multiplyElementWise(CB, CC);
		//Matrix CC = CA.multiplyElementWise(CB);
#ifdef DEBUG
		CA.print();
		std::cout << std::endl;
		CB.print();
		std::cout << std::endl;
		CC.print();
		std::cout << std::endl;
#endif
		Matrix::forceUseGPU();
		Matrix A;
		A.copyToThis(CA);
		Matrix B;
		B.copyToThis(CB);
		Matrix C(A.height(), A.width());
		A.multiplyElementWise(B, C);
		//Matrix C = A.multiplyElementWise(B);
#ifdef DEBUG
		C.print();
#endif
		std::cout << "Comparison: " << Matrix::compare(CC, C, -100, 100) << std::endl << std::endl;
	}
#define DEBUG
	void sigmoid() {
		std::cout << "---------- Sigmoid ----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(4, 3);
		CA.randomFill(-10, 10);
#ifdef DEBUG
		CA.print();
		std::cout << std::endl;
#endif
		Matrix CC = CA.sigmoid();
#ifdef DEBUG
		CC.print();
		std::cout << std::endl;
#endif
		Matrix::forceUseGPU();

		Matrix A;
		A.copyToThis(CA);
		Matrix C = A.sigmoid();
#ifdef DEBUG
		C.print();
#endif
		std::cout << "Comparison: " << Matrix::compare(CC, C, 0, 1) << std::endl << std::endl;


		std::cout << "---------- Sigmoid Differential----------" << std::endl;
		Matrix::forceUseCPU();
#ifdef DEBUG
		CC.print();
		std::cout << std::endl;
#endif
		CA = CC.sigmoidDifferential();
#ifdef DEBUG
		CA.print();
		std::cout << std::endl;
#endif
		Matrix::forceUseGPU();
		A.copyToThis(CC);
#ifdef DEBUG
		A.print();
		std::cout << std::endl;
#endif
		C = A.sigmoidDifferential();
#ifdef DEBUG
		C.print();
#endif
		std::cout << "Comparison: " << Matrix::compare(CA, C, -10, 10) << std::endl << std::endl;
		
	}
#undef DEBUG
	void add() {
		std::cout << "---------- Add ----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(4, 3);
		CA.randomFill(-10, 10);
		Matrix CB(4, 3);
		CB.randomFill(-10, 10);
		Matrix CC = CA.add(CB);
#ifdef DEBUG
		CA.print();
		std::cout << std::endl;
		CB.print();
		std::cout << std::endl;
		CC.print();
		std::cout << std::endl;
#endif
		Matrix::forceUseGPU();
		Matrix A;
		A.copyToThis(CA);
		Matrix B;
		B.copyToThis(CB);
		Matrix C = A.add(B);
#ifdef DEBUG
		C.print();
#endif
		std::cout << "Comparison: " <<  Matrix::compare(CC, C, -20, 20) << std::endl << std::endl;
	}

	void addAssign() {
		std::cout << "---------- Add Assign ----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(4, 3);
		CA.randomFill(-10, 10);
		Matrix CC;
		CC.copyToThis(CA);
		Matrix CB(4, 3);
		CB.randomFill(-10, 10);
		CA.addAssign(CB);
#ifdef DEBUG
		CA.print();
		std::cout << std::endl;
#endif
		Matrix::forceUseGPU();
		Matrix A;
		A.copyToThis(CC);
		Matrix B;
		B.copyToThis(CB);
		A.addAssign(B);
#ifdef DEBUG
		A.print();
#endif
		std::cout << "Comparison: " << Matrix::compare(CA, A, -20, 20) << std::endl << std::endl;
	}

	void subtract() {
		std::cout << "---------- Subtract ----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(4, 3);
		CA.randomFill(-10, 10);
		Matrix CB(4, 3);
		CB.randomFill(-10, 10);
		Matrix CC = CA.subtract(CB);
#ifdef DEBUG
		CA.print();
		std::cout << std::endl;
		CB.print();
		std::cout << std::endl;
		CC.print();
		std::cout << std::endl;
#endif
		Matrix::forceUseGPU();
		Matrix A;
		A.copyToThis(CA);
		Matrix B;
		B.copyToThis(CB);
		Matrix C = A.subtract(B);
#ifdef DEBUG
		C.print();
#endif
		std::cout << "Comparison: " << Matrix::compare(CC, C, -20, 20) << std::endl << std::endl;
	}

	void subtractAssign() {
		std::cout << "---------- Subtract Assign ----------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(4, 3);
		CA.randomFill(-10, 10);
		Matrix CC;
		CC.copyToThis(CA);
		Matrix CB(4, 3);
		CB.randomFill(-10, 10);
		CA.subtractAssign(CB);
#ifdef DEBUG
		CA.print();
		std::cout << std::endl;
#endif
		Matrix::forceUseGPU();
		Matrix A;
		A.copyToThis(CC);
		Matrix B;
		B.copyToThis(CB);
		A.subtractAssign(B);
#ifdef DEBUG
		A.print();
#endif
		std::cout << "Comparison: " << Matrix::compare(CA, A, -20, 20) << std::endl << std::endl;
	}

	void addConst() {
		std::cout << "------- Add Const -------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(4, 3);
		CA.randomFill(-10, 10);
#ifdef DEBUG
		CA.print();
		std::cout << std::endl;
#endif
		Matrix CC = CA.addConst(3);
#ifdef DEBUG
		CC.print();
		std::cout << std::endl;
#endif
		Matrix::forceUseGPU();

		Matrix A;
		A.copyToThis(CA);
		Matrix C = A.addConst(3);

#ifdef DEBUG
		C.print();
#endif
		std::cout << "Comparison: " << Matrix::compare(CC, C, -7, 13) << std::endl << std::endl;
	}

	void copyTest(){
		std::cout << "------- Copy test -------" << std::endl;
		Matrix::forceUseCPU();
		Matrix i(4,3);
		i.randomFill(-100,100);
		Matrix a(4,3);
		a.copyToThis(i);
		std::cout << "Comparison: " << Matrix::compare(i, a, -100, 100) << std::endl << std::endl;

		Matrix::forceUseGPU();
		Matrix b(4,3);
		b.copyToThis(i);
		std::cout << "Comparison: " << Matrix::compare(i, b, -100, 100) << std::endl << std::endl;
	}

	void scale() {
		std::cout << "------- Scale -------" << std::endl;
		Matrix::forceUseCPU();
		Matrix CA(4, 3);
		CA.randomFill(-10, 10);
#ifdef DEBUG
		CA.print();
		std::cout << std::endl;
#endif
		Matrix CC = CA.scale(3);
#ifdef DEBUG
		CC.print();
		std::cout << std::endl;
#endif
		Matrix::forceUseGPU();

		Matrix A;
		A.copyToThis(CA);
		Matrix C = A.scale(3);
#ifdef DEBUG
		C.print();
#endif

		std::cout << "Comparison: " << Matrix::compare(CC, C, -30, 30) << std::endl << std::endl;
	}

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
		

	}

	void testAll() {
		copyTest();
		trans();
		transMult();
		mult();
		multMany();
		multManySmall();
		multElementWise();
		multElementMany();
		sigmoid();
		add();
		addAssign();
		subtract();
		subtractAssign();
		addConst();
		scale();
	}

}