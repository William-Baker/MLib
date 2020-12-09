#pragma once
#include "Neural.hpp"
#include <vector>
#include <future>
#include <AsyncJob.hpp>
#include <png.hpp>
#include "mnist//mnist_reader.hpp"

#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock


//TODO look at weight decay

class DataSetProvider{
	public:
	virtual void initialise(std::string file_path) = 0;
	virtual MLCase get_next_case() = 0;
	virtual MLCase get_current_case() = 0;
	virtual void randomise_positions() = 0;
	virtual size_t get_size() = 0;
};

class Trainer{
	public:
	DataSetProvider* data = 0;
	NeuralNetwork* NN = 0;
	size_t batch_size = 0; //The number of of cases to train on before checking the performance across those cases

	/**
	 * @param batch_size set to 0 to calculate batch size automatically
	 */
	Trainer(DataSetProvider* data, NeuralNetwork* NN, size_t batch_size = 0){
		this->batch_size = batch_size ? batch_size : data->get_size();
		this->data = data;
		this->NN = NN;
	}
	/**
	 * @param performance_target the desired accuracy of the network across a batch (0-1) 0 is better
	 * @return performance of the network (0-1) 0 is better
	 */
	double beginTraining(double performance_target, double LR = 0.05){
		double current_performance = 1;

		//>= so 0 is a valid target
		while(current_performance >= performance_target){
			double performance_tally = 0;
			for(size_t batch_iteration = 0; batch_iteration < batch_size; batch_iteration ++){
				MLCase current_case = data->get_next_case();
				NN->compute(current_case.inputs);
				Matrix tmp(current_case.outputs, false);
				performance_tally += NN->backprop(tmp, LR);
			}
			current_performance = performance_tally / batch_size;
			//std::cout << "Current performance: " << current_performance << std::endl;
			data->randomise_positions();
			//NN->print();
		}
		return current_performance;
	}

};

class XOR : public DataSetProvider{
	public:
	std::vector<MLCase> cases;
	size_t ptr = 0;
	

	void initialise(std::string file_path) override{
		Matrix* oo_i = new Matrix(2,1);
		oo_i->setIndex(0,0.1);
		oo_i->setIndex(1,0.1);
		Matrix* oo_o = new Matrix(1,1);
		oo_o->setIndex(0,0);
		cases.push_back(MLCase(oo_i->getStrategy(), oo_o->getStrategy()));

		Matrix* on_i = new Matrix(2,1);
		on_i->setIndex(0,0.1);
		on_i->setIndex(1,1);
		Matrix* on_o = new Matrix(1,1);
		on_o->setIndex(0,1);
		cases.push_back(MLCase(on_i->getStrategy(), on_o->getStrategy()));

		Matrix* no_i = new Matrix(2,1);
		no_i->setIndex(0,1);
		no_i->setIndex(1,0.1);
		Matrix* no_o = new Matrix(1,1);
		no_o->setIndex(0,1);
		cases.push_back(MLCase(no_i->getStrategy(), no_o->getStrategy()));

		Matrix* nn_i = new Matrix(2,1);
		nn_i->setIndex(0,1);
		nn_i->setIndex(1,1);
		Matrix* nn_o = new Matrix(1,1);
		nn_o->setIndex(0,0);
		cases.push_back(MLCase(nn_i->getStrategy(), nn_o->getStrategy()));

		//Set 2

		oo_i = new Matrix(2,1);
		oo_i->setIndex(0,0.4);
		oo_i->setIndex(1,0.6);
		oo_o = new Matrix(1,1);
		oo_o->setIndex(0,0);
		cases.push_back(MLCase(oo_i->getStrategy(), oo_o->getStrategy()));

		on_i = new Matrix(2,1);
		on_i->setIndex(0,0.4);
		on_i->setIndex(1,0.6);
		on_o = new Matrix(1,1);
		on_o->setIndex(0,1);
		cases.push_back(MLCase(on_i->getStrategy(), on_o->getStrategy()));

		no_i = new Matrix(2,1);
		no_i->setIndex(0,0.6);
		no_i->setIndex(1,0.4);
		no_o = new Matrix(1,1);
		no_o->setIndex(0,1);
		cases.push_back(MLCase(no_i->getStrategy(), no_o->getStrategy()));

		nn_i = new Matrix(2,1);
		nn_i->setIndex(0,0.6);
		nn_i->setIndex(1,0.6);
		nn_o = new Matrix(1,1);
		nn_o->setIndex(0,0);
		cases.push_back(MLCase(nn_i->getStrategy(), nn_o->getStrategy()));
	}

	MLCase get_next_case() override {
		ptr = (ptr+1)%4;
		return cases[ptr];
	}

	MLCase get_current_case() override {
		return cases[ptr];
	}

	void randomise_positions() override {
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

  		shuffle (cases.begin(), cases.end(), std::default_random_engine(seed));
	}

	size_t get_size() override{
		return 4;
	}

};



class TrainCases {
	
	public:

	std::vector<MLCase> live_cases;
	std::vector<MLCase> dead_cases;
	MLCase current;

	bool dead_storage = false;
	double max_GPU_memory_usage;
	bool using_gpu = false;

	long size;
	long ptr;
	long livePtr;
	inline long deadPtr(){return ptr-livePtr;}
	inline void next_live(){ptr++; ptr%=size; livePtr++; livePtr%=live_cases.size();}

	std::future<void> future;

	/**
	 * @param max_GPU_memory_usageIn percentage of allowable VRam usage
	 */
	TrainCases(double max_GPU_memory_usageIn) {
		size = 0;
		ptr = 0;
		livePtr = 0;
		max_GPU_memory_usage = max_GPU_memory_usageIn;
		using_gpu = Matrix::checkGPU();
	}

	void load_case(MLCase this_case) {
		if(dead_storage){
			dead_cases.push_back(this_case);
		}
		else{
			live_cases.push_back(this_case);
		}
		size++;
	}

	
	/**
	 * curerntly only 2D data structures are supported
	 * dimensions are x, y, z, alpha, beta
	 */
	void load_data(double* data, std::vector<size_t> dim_data, double* label, std::vector<size_t> dim_label){
		switchToCPUOnUsage(max_GPU_memory_usage);
		AbstractMatrix<double>* data_matrix;
		AbstractMatrix<double>* label_matrix;
		if(using_gpu){
			data_matrix = new GPUMatrix(dim_data[1], dim_data[0], data);
			label_matrix = new GPUMatrix(dim_label[1], dim_label[0], label);
		}
		else {
			data_matrix = new GPUMatrix(dim_data[1], dim_data[0], data);
			label_matrix = new GPUMatrix(dim_label[1], dim_label[0], label);
		}

		load_case( MLCase(data_matrix, label_matrix) );
	}

	void load_dataset(std::string subfolder, MLStruct<double>* input, MLStruct<double>* output, double maxMemUsage) {
		mnist::MNIST_dataset<std::vector, png::tRNS, uint8_t> dataset = mnist::read_dataset(subfolder, 0, 0); //read mnists data set folder with TODO: "filename" name
		for(long index = 0; index < dataset.test_images.size(); index ++){
			
			//TODO move all this to a dedicated load case class taking an array, its dimensions and the label array + dimensions
			// Matrix* test_label = new Matrix(1,1);					//Retrieve label and assign to matrix
			// test_label->setIndex(0, dataset.test_labels[index]);
			
			// std::vector<uint8_t> & test_image = dataset.test_images[index]; //reference image in dataset
			// size_t data_dimension = sqrt(test_image.size());
			// Matrix* data = new Matrix(data_dimension, data_dimension, test_image.data());
			
			
			// MLCase* case_load = new MLCase(data->getStrategy(), test_label->getStrategy());
			// test_image.~vector();
			//switchToCPUOnUsage(maxMemUsage);
		}


		Matrix::forceUseGPU();
	}



	public:
	


	void next_case() {
		if (dead_cases.size()) {
			future = std::async(std::launch::async, MLCase::swap, &live_cases[ptr], &dead_cases[deadPtr()]);
		}
		next_live();
	}

	class ImageContainer {
	public:
		uint8_t* arr = 0;
		ImageContainer(uint8_t* arr) {
			this->arr = arr;
		}
		long getW() {
			return arr[0];
			;
		}
		long getH() {
			return arr[0];
			;
		}

	};
private:
	void switchToCPUOnUsage(double maxMemUsage) {
		if (Matrix::checkGPU()) {
			size_t free, used;
			cudaMemGetInfo(&free, &used);
			if (used / (used + free) > maxMemUsage) {
				using_gpu = false;
				dead_storage = true;
			}
			else using_gpu = true;//we may want to resume GPU usage
		}
	}
};

class TrestManager{
    std::vector<std::vector<uint8_t>> array;
     std::vector<double> labels;
    TrestManager(std::vector<std::vector<uint8_t>> array, std::vector<double> labels ){
        this->array = array;
        this->labels = labels;

    }
};

/**
 * @param arr the array of data to write to an imaghe
 * @param x horzontal bits
 * @param y vertical bits
 * @param d the number of bits per pixel
 */
void imageOut(uint8_t* arr, int x, int y, int d){
	
    static int id = 0;
	class ImageWriteJob{
		public:
		uint8_t* arr;
		int x;
		int y;
		int d;
		ImageWriteJob(uint8_t* a, int xp, int yp, int dp){
			arr = a;
			x = xp;
			y = yp;
			d = dp;
		}
		static int write(ImageWriteJob* job, void* user){
			int id = *(int*)user;
			if(job->d == 24){
				long i = 0;
				png::image< png::rgb_pixel > image(job->x, job->y);
				for (png::uint_32 y = 0; y < image.get_height(); ++y)
				{
					for (png::uint_32 x = 0; x < image.get_width(); ++x)
					{
						image[y][x] = png::rgb_pixel(job->arr[i], job->arr[i+1], job->arr[i+2]);
						i += 3;
					}
				}
				image.write(std::to_string(id) + (std::string)".png");
			}
			else{
				double i = 0;
				png::image< png::rgb_pixel > image(job->x, job->y);
				for (png::uint_32 y = 0; y < image.get_height(); ++y)
				{
					for (png::uint_32 x = 0; x < image.get_width(); ++x)
					{
						i += job->d / 8;
						long index = std::floor(i);
						uint64_t val = ((uint64_t*)(job->arr))[index];
						val = val >> (int)std::floor((i-index)*8);
						val = val & ((int)pow(2,job->d))-1;
						double pixel = (val * 255) /pow(2, job->d);//compress the dynamic range of d bits to 8 bit and gray scale across spectrum
						image[y][x] = png::rgb_pixel(pixel, pixel, pixel); 
					}
				}
				image.write(std::to_string(id) + (std::string)".png");
			}
			return 0;

			
		}
	};
	static AsyncJob<ImageWriteJob> imageWriter(ImageWriteJob::write, &id);
	imageWriter.addJob(new ImageWriteJob(arr, x, y, d));
    
    

    id++;
}