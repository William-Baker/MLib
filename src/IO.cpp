#include <IO.hpp>
#include <png.hpp>
#include <AsyncJob.hpp>

/* typedef std::ostream& (*manip) (std::ostream&);

class console {

};
//put it in a cpp
template <class T> console& operator<< (console& con, const T& x) {
    #ifdef __VERBOSE__
    std::cout << x;
    #endif
    return con;
}
template <class T> console& operator>>(console& con,  T& x) {
    #ifdef __VERBOSE__
    std::cin >>x;
    #endif
    return con;
}
console& operator<< (console& con, manip manipulator){
    #ifdef __VERBOSE__
    std::cout<<manipulator;
    #endif
    return con;
} */


void IO::IO_log(LOG_TYPE log_type, std::string message, std::string function_name){
   const char* lookup[] = {
      "ERROR       ",
      "WARNING     ",
      "INFO        ",
      "FATAL ERROR "
   };
   fprintf(stderr, "%s", (lookup[log_type] + message + "\n        - " + function_name + "\n").c_str());
   if(log_type == FATAL_ERROR) throw(message);
}


std::chrono::time_point<std::chrono::high_resolution_clock> Timer::startTime;
void Timer::start() {
	startTime = std::chrono::high_resolution_clock::now();
}
size_t Timer::time() {
	auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
	return std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
}


/**
 * @param arr the array of data to write to an imaghe
 * @param x horzontal bits
 * @param y vertical bits
 * @param d the number of bits per pixel
 */
void write_image_to_file(uint8_t* arr, int x, int y, int d){
	
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
			else if(job->d == 8){
				auto index = 0;
				png::image< png::rgb_pixel > image(job->x, job->y);
				for (auto y = 0; y < image.get_height(); ++y)
				{
					for (auto x = 0; x < image.get_width(); ++x)
					{
						image[y][x] = png::rgb_pixel(job->arr[index],job->arr[index],job->arr[index]);
						index++;
					}
				}
				image.write(std::to_string(id) + (std::string)".png");
			}
			else {
				ilog(ERROR, "unsupported bit depth");
				return 1;
			}
			return 0;

			
		}
	};
	static AsyncJob<ImageWriteJob> imageWriter(ImageWriteJob::write, &id);
	imageWriter.addJob(new ImageWriteJob(arr, x, y, d));
    
    

    id++;
}