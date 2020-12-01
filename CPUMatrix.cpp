
#include "CPUMatrix.hpp"
#include "Matrix.hpp"



//=========================================================== CPU Matrix ==========================================================================

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
//TODO remove unactivated output
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
					
					//out->setIndex(oY * outZ + oZ , oX, tanh(temp));
					out->setIndex(oY + outZ * oZ , oX, tanh(temp));
				}
			}
		}
	
	}

/**
 * @param in input matrix y: Y*Z, x: X
 * @param layer convolution matrix y: convY*Z, x: convX1 + convX2 + convX3... convX(convZ) - the Z dimension are stored adjacently in the Y axis, The convZ dimension are split into chunks in the X axis
 * @param this_layer_conv_error the error in this conv layer (LR already applied)
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
void CPUMatrix::convBackprop(AbstractMatrix* in, AbstractMatrix* layer, AbstractMatrix* this_layer_conv_error, AbstractMatrix* prevError, AbstractMatrix* bias, AbstractMatrix* out, AbstractMatrix* out_error, AbstractMatrix* gradient, int outY, int outX, int outZ, int convY, int convX, int convZ, double LR) {
	for (int x = 0; x < out->get_size(); x++) {
		gradient->setIndex(x, out_error->index(x) * tanhd_on_tanh(out->index(x)));

	}

	//Fill our matrices that dont get overwitten
	prevError->fill(0);
	this_layer_conv_error->fill(0);

	for (int oZ = 0; oZ < outZ; oZ++) {
		for (int oX = 0; oX < outX; oX++) {
			for (int oY = 0; oY < outY; oY++) {
			
				double this_conv_output_gradient = gradient->index(oY*outZ + oZ, oX);

				
				for (int cX = 0; cX < convX; cX++) {
					for (int cYZ = 0; cYZ < convY * convZ; cYZ++) {

						double error_at_index_in_conv = in->index(oY+cYZ, oX+cX) * this_conv_output_gradient;
						
						
						
						prevError->addIndex(oY+cYZ, oX+cX, layer->index(cYZ, cX + convX*oZ) * this_conv_output_gradient);

						error_at_index_in_conv *= LR;
						this_layer_conv_error->addIndex(cYZ, cX + convX*oZ, error_at_index_in_conv);
					}
				}
			}
		}
	}

	layer->addAssign(this_layer_conv_error);
	
}


//=================================================================================================================================================

