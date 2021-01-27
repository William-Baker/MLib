#include "ConvolutionalLayer.hpp"
#include "FeedForwardLayer.hpp"

void ConvolutionalLayer::allocate() {
    layer = Matrix(convY * inZ, convX * outZ);
    bias = Matrix(outZ, 1);
    output = Matrix(outY * outZ, outX);
    layer_deltas = Matrix(convY * inZ, convX * outZ);
    gradient = Matrix(outY * outZ, outX);
    errorSignal = Matrix(inY * inZ, inX);
}

void ConvolutionalLayer::randomise() {
    layer.randomFill(-0.3, -0.05, 0.05, 0.3);
    bias.randomFill(-0.3, -0.05, 0.05, 0.3);
}

void ConvolutionalLayer::zeros() {
    layer.fill(0);  
    bias.fill(0);
    output.fill(0);
    layer_deltas.fill(0);
    gradient.fill(0);
    errorSignal.fill(0);
}

void ConvolutionalLayer::compute(MLStruct<double>* inputIn) {
    delete input;
    input = new Matrix(inputIn, false);
    compute();
}

void ConvolutionalLayer::compute() {
    //AbstractMatrix* layer, AbstractMatrix* bias, AbstractMatrix* unactivated_output, AbstractMatrix* out
    input->convolute(layer,  bias, output, outY, outX, outZ, convY, convX, inZ);
    if (get_next_layer()) {
        get_next_layer()->compute();
    }
}
// void compute() {
// 	compute(get_prev_layer()->get_output());
// }


void ConvolutionalLayer::backprop(Matrix &outError, double LR) {	
    outError.convBackprop(*input, layer, layer_deltas, errorSignal, bias, output, outError, gradient, outY, outX, outZ, convY, convX, inZ, LR);

    if (get_prev_layer()) {
        get_prev_layer()->backprop(LR);
    }

}
void ConvolutionalLayer::backprop(double LR) {
    backprop(*error_signal_next_layer, LR);
}
void ConvolutionalLayer::backprop(MLStruct<double>* err, double LR) {
    Matrix tmp(err, false);
    backprop(tmp, LR);
}

//Matrix layer;
//Matrix bias;
//Matrix output;
//Matrix net;
//Matrix errorSignal;//The Error Signal at teh input
void ConvolutionalLayer::print() {
    std::cout << "--------------- CV Conv Layer Y: " << convY << " X: " << convX << " Z: " << inZ << "---------------" << std::endl;
    layer.print();
    std::cout << "--------------- CV Conv Layer Delta Y: " << convY << " X: " << convX << " Z: " << inZ << "---------------" << std::endl;
    layer_deltas.print();
    std::cout << "--------------- CV Bias Layer Y: " << bias.height() << " ---------------" << std::endl;
    bias.print();
    std::cout << "--------------- CV output Layer Y: " << outY << " X: " << outX << " Z: " << outZ << " ---------------" << std::endl;
    output.print();
    std::cout << "--------------- CV error Layer Y: " << outY << " X: " << outX << " Z: " << outZ << " ---------------" << std::endl;
    errorSignal.print();
}

void ConvolutionalLayer::connect_to_next_layer(Layer* next){
    if(dynamic_cast<OutputLayer*>(next)){
        OutputLayer* layer = static_cast<OutputLayer*>(next);
        error_signal_next_layer = new Matrix(Matrix(layer->get_error_signal(), false).copy_keeping_same_data());
        error_signal_next_layer->resize(outY * outZ, outX);
    }
    else if(dynamic_cast<FeedForwardLayer*>(next)){
        FeedForwardLayer* layer = static_cast<FeedForwardLayer*>(next);
        error_signal_next_layer = new Matrix(Matrix(layer->get_error_signal(), false).copy_keeping_same_data());
        error_signal_next_layer->resize(outY * outZ, outX);
    }
    else if(dynamic_cast<ConvolutionalLayer*>(next)){
        ConvolutionalLayer* layer = static_cast<ConvolutionalLayer*>(next);
        error_signal_next_layer = new Matrix(Matrix(layer->get_error_signal(), false).copy_keeping_same_data());
        error_signal_next_layer->resize(outY * outZ, outX);
    }
    else{
        ilog(FATAL_ERROR, "unrecognized next layer");
    }
}

void ConvolutionalLayer::ConvolutionalLayer::connect_to_prev_layer(Layer* prev){
    if(dynamic_cast<InputLayer*>(prev)){
        InputLayer* layer = static_cast<InputLayer*>(prev);
    }
    else if(dynamic_cast<FeedForwardLayer*>(prev)){
        FeedForwardLayer* layer = static_cast<FeedForwardLayer*>(prev);
        input = new Matrix(Matrix(layer->get_output(), false).copy_keeping_same_data());
        input->resize(inY * inZ, inX); //use error signal to resize

    }
    else if(dynamic_cast<ConvolutionalLayer*>(prev)){
        ConvolutionalLayer* layer = static_cast<ConvolutionalLayer*>(prev);
        input = new Matrix(Matrix(layer->get_output(), false).copy_keeping_same_data());
        input->resize(inY * inZ, inX);
    }
    else{
        ilog(FATAL_ERROR, "unrecognized previous layer");
    }
}