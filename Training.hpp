#include "Neural.hpp"
#include <vector>
#include <future>
#include <AsyncJob.hpp>
#include <png.hpp>
#include "mnist//mnist_reader.hpp"

class TrainCases {
	std::vector<MLCase> casesLive;
	std::vector<MLCase> casesDead;
	MLCase current;

	long size;
	long ptr;
	long livePtr;

	std::future<void> future;

	TrainCases() {
		size = 0;
		ptr = 0;
		livePtr = 0;
		
		
	}

	void load_case(MLCase mlCase) {
		/*casesLive.resize(casesLive.size() + 1);
		casesLive[casesLive.size() - 1] = mlCase;*/
		casesLive.push_back(mlCase);
		size++;
	}

	void load_case(MLStruct<double>* input, MLStruct<double>* output) {
		MLCase trainingCase;
		trainingCase.inputs = input;
		trainingCase.outputs = output;
		load_case(trainingCase);
		
	}
	void load_dataset(std::string subfolder, MLStruct<double>* input, MLStruct<double>* output, double maxMemUsage) {
		mnist::MNIST_dataset<std::vector, png::tRNS, uint8_t> dataset = mnist::read_dataset(subfolder, 0, 0); //read mnists data set folder with TODO: "filename" name
		for(long index = 0; index < dataset.test_images.size(); index ++){
			std::vector<uint8_t> test_image = dataset.test_images[index];
			//std::vector<uint8_t> test_label = dataset.test_labels[index];
			//auto test_label = dataset.test_labels[index];
			Matrix* test_label = new Matrix(1,1);
			test_label->setIndex(0, dataset.test_labels[index]);
			Matrix* data = new Matrix((int)sqrt(test_image.size()), (int)sqrt(test_image.size()));
			data->copyToThis(test_image.data());
			MLCase* case_load = new MLCase((MLStruct<double>*)data, (MLStruct<double>*)test_label);
			test_image.~vector();
			switchToCPUOnUsage(maxMemUsage);
		}


		Matrix::forceUseGPU();
	}
	void next_case() {
		int oldLivePtr = ptr;
		ptr++;
		ptr %= size;
		livePtr++;
		livePtr %= casesLive.size();
		if (casesDead.size()) {
			future = std::async(std::launch::async, MLCase::swap, &casesLive[ptr - livePtr], &casesLive[oldLivePtr]);
		}
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
		if (Matrix::hasGPU()) {
			size_t free, used;
			cudaMemGetInfo(&free, &used);
			if (used / used + free > maxMemUsage) {
				Matrix::forceUseCPU();
			}
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