
#include "CPUMatrix.hpp"
#include "GPUMatrix.hpp"
#include "Matrix.hpp"



//=========================================================== CPU Matrix ==========================================================================

CPUMatrix::CPUMatrix(const AbstractMatrix<double>* src) : CPUMatrix(){
	if(dynamic_cast<const CPUMatrix*>(src)){
		const CPUMatrix* actual = static_cast<const CPUMatrix*>(src);
		transfer(actual->copy());
	}
	else if(dynamic_cast<const GPUMatrix*>(src)){
		const GPUMatrix* actual = static_cast<const GPUMatrix*>(src);
		x = actual->x;
		y = actual->y;
		size = actual->get_size();
		arr = allocate_CPU_memory<double>(size);
		copy_GPU_memory<double>(arr, actual->arr, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}
	else{
		ilog(FATAL_ERROR, "unknown source for copy constructor");
	}
}











//Functionality

void CPUMatrix::randomFill(double min, double max) {
	std::uniform_real_distribution<double> RandDist(min, max);
	for (int index = 0; index < size; index++) {
		arr[index] = RandDist(Matrix::mt);
	}
};


void CPUMatrix::randomFill(double lowerMin, double lowerMax, double upperMin, double upperMax) {
	std::uniform_real_distribution<double> SignRandDist(0, 2);
	std::uniform_real_distribution<double> RandDistLower(lowerMin, lowerMax);
	std::uniform_real_distribution<double> RandDistUpper(upperMin, upperMax);
	for (int index = 0; index < size; index++) {
		if (SignRandDist(Matrix::mt) > 1) {
			arr[index] = RandDistLower(Matrix::mt);
		}
		else {
			arr[index] = RandDistUpper(Matrix::mt);
		}
	}
}
//TODO remove unactuated output - i think it already has been?
//TODO the error is here!
void CPUMatrix::convolute(AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* out, int outY, int outX, int outZ, int convY, int convX, int convZ) {
	for (int oZ = 0; oZ < outZ; oZ++) {
		for (int oX = 0; oX < outX; oX++) {
			for (int oY = 0; oY < outY; oY++) {
				double temp = 0;
				for (int cX = 0; cX < convX; cX++) {
					for (int cYZ = 0; cYZ < convY * convZ; cYZ++) {
								//temp += index(oY*inZ + cYZ, oX + cX) * layer->index(cYZ, cX + oZ * convX);
								temp += index(oY + cYZ, oX + cX) * layer->index(cYZ, cX + convX*oZ);
						}
					}
					temp += bias->index(oZ);
					if(isnan(out->index(oY + outZ * oZ , oX))){
						std::cout << "A\n";
					}
					//out->setIndex(oY * outZ + oZ , oX, tanh(temp));
					out->setIndex(oY + outZ * oZ , oX, tanh(temp));
					if(isnan(out->index(oY + outZ * oZ , oX))){
						std::cout << "A\n";
					}
				}
			}
		}
	
	}

/**
 * this - output error to back propigate
 * @param input matrix y: Y*Z, x: X
 * @param layer convolution matrix y: convY*Z, x: convX1 + convX2 + convX3... convX(convZ) - the Z dimension are stored adjacently in the Y axis, The convZ dimension are split into chunks in the X axis
 * @param layer_deltas the error in this conv layer (LR already applied)
 * @param bias size = convZ
 * @param prevError error at the input to the layer
 * @param out the output of the network
 * @param out_error error at the output of this layer
 * @param gradient, the gradient at the output of this layer
 * @param LR learning rate scalar to apple
 * @param outY the Y size of the output matrix = inY - floor(convY/2)-1
 * @param outX the X size of the output matrix = inX - floor(convX/2)-1
 * @param outZ the Z depth of the ouput eqault to the number of conv filters, also called f
 * @param convX the X dimension of the convolution layer
 * @param convY the Y dimension of the convolution layer
 * @param convZ the Z depth of the convolution layer, equal to the Z dimension of the input (the Z dimension of the input can be used as RGB or whatever)
 */
void CPUMatrix::convBackprop(AbstractMatrix* input, AbstractMatrix* layer, AbstractMatrix* layer_deltas, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* out, AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) {
	for (int x = 0; x < out->get_size(); x++) {
		gradient->setIndex(x, index(x) * tanhd_on_tanh(out->index(x)));

	}

	//Fill our matrices that dont get overwitten
	prevError->fill(0);
	layer_deltas->fill(0);

	for (int oZ = 0; oZ < outZ; oZ++) {
		for (int oX = 0; oX < outX; oX++) {
			for (int oY = 0; oY < outY; oY++) {
			
				double this_conv_output_gradient = gradient->index(oY*outZ + oZ, oX);

						
				
				for (int cX = 0; cX < convX; cX++) {
					for (int cYZ = 0; cYZ < convY * convZ; cYZ++) {
						
						double error_at_index_in_conv = input->index(oY+cYZ, oX+cX) * this_conv_output_gradient;
						
						
						
						prevError->addIndex(oY+cYZ, oX+cX, layer->index(cYZ, cX + convX*oZ) * this_conv_output_gradient);

						error_at_index_in_conv *= LR;
						layer_deltas->addIndex(cYZ, cX + convX*oZ, error_at_index_in_conv);
					}
				}
			}
		}
	}

	layer->addAssign(layer_deltas);
	
}


//=================================================================================================================================================

