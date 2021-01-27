#include "FeedForwardLayer.hpp"
#include "ConvolutionalLayer.hpp"


void FeedForwardLayer::randomise() {
    weights.randomFill(-0.3, -0.05, 0.05, 0.3);
    biases.randomFill(-0.3, -0.05, 0.05, 0.3);
    //weights.randomFill(0, 1);
    //biases.randomFill(0, 1);
}

void FeedForwardLayer::zeros(){
    weights.fill(0);
    weights_deltas.fill(0);
    biases.fill(0);
    output.fill(0);
    error_signal.fill(0);
    net.fill(0);
    temp2.fill(0);
    gradient.fill(0);
}

// void compute() {
// 	Matrix tmp(get_prev_layer()->get_output(), false);
// 	compute(tmp);
// }

void FeedForwardLayer::compute(MLStruct<double>* inputIn) {
    delete input;
    input = new Matrix(inputIn, false);
    compute();
}
void FeedForwardLayer::compute() {
    weights.multiply(*input, output);
    output.add(biases, net);
    net.sigmoid(output);
    if (get_next_layer() != 0) {
        get_next_layer()->compute();
    }
}

void FeedForwardLayer::calculateErrorSignal(Matrix& outputError) {
    weights.multiplyA(outputError, error_signal);
    /*std::cout << "------- Output Error ------------" << std::endl;
    outputError.print();
    std::cout << "------- Weights ------------" << std::endl;
    weights.print();
    std::cout << "------- Error Signal ------------" << std::endl;
    errorSig.print();
    std::cout << std::endl;
    std::cout << std::endl;
    */
}

void FeedForwardLayer::update(double LR, Matrix& errorSigAtOutput) {
    output.sigmoidDifferential(gradient);
    gradient.scale(LR, temp2);
    temp2.multiplyElementWise(errorSigAtOutput, gradient);
    gradient.multiplyB(*input, weights_deltas);
    weights.addAssign(weights_deltas);
    //std::cout << "------- Weights Deltas ------------" << std::endl;
    //weights_deltas.print();
    biases.addAssign(gradient);
}

void FeedForwardLayer::backprop(double LR) {
    calculateErrorSignal(*error_signal_next_layer);
    update(LR, *error_signal_next_layer);
    get_prev_layer()->backprop(LR);	
}
/**
 * @deprecated
 */
void FeedForwardLayer::backprop(MLStruct<double>* err, double LR) {
    ilog(WARNING, "backprop call deprecated, should connect to next layer instead");
    Matrix tmp(err,false);
    backprop(tmp, LR);
}
/**
 * @deprecated
 */
void FeedForwardLayer::backprop(Matrix& outErrorSig, double LR)  {
    ilog(WARNING, "backprop call deprecated, should connect to next layer instead");
    calculateErrorSignal(outErrorSig);
    update(LR, outErrorSig);
    get_prev_layer()->backprop(LR);	
}



void FeedForwardLayer::print() {

    std::cout << "--------------- FF Weights Layer Y: " << weights.height() << " X: " << weights.width() <<  " ---------------" << std::endl;
    weights.print();
    std::cout << "--------------- FF Weights Deltas Layer Y: " << weights_deltas.height() << " X: " << weights_deltas.width() << " ---------------" << std::endl;
    weights_deltas.print();
    std::cout << "--------------- FF Biases Layer Y: " << biases.height() << " ---------------" << std::endl;
    biases.print();
    std::cout << "--------------- FF Output Layer Y: " << output.height() << " ---------------" << std::endl;
    output.print();
    std::cout << "--------------- FF Out Error Layer Y: " << error_signal.height() << " ---------------" << std::endl;
    error_signal.print();
    
}


void FeedForwardLayer::connect_to_next_layer(Layer* next){
    if(dynamic_cast<OutputLayer*>(next)){
        OutputLayer* layer = static_cast<OutputLayer*>(next);
        error_signal_next_layer = new Matrix(layer->get_error_signal(), false);
        error_signal_next_layer->resize(biases.size(),1);
    }
    else if(dynamic_cast<FeedForwardLayer*>(next)){
        FeedForwardLayer* layer = static_cast<FeedForwardLayer*>(next);
        error_signal_next_layer = new Matrix(layer->get_error_signal(), false);
        error_signal_next_layer->resize(biases.size(),1);
    }
    else if(dynamic_cast<ConvolutionalLayer*>(next)){
        ConvolutionalLayer* layer = static_cast<ConvolutionalLayer*>(next);
        error_signal_next_layer = new Matrix(Matrix(layer->get_error_signal(), false).copy_keeping_same_data());
        error_signal_next_layer->resize(biases.size(),1);
    }
    else{
        ilog(FATAL_ERROR, "unrecognized next layer");
    }
}

void FeedForwardLayer::connect_to_prev_layer(Layer* prev){
    if(dynamic_cast<InputLayer*>(prev)){
        InputLayer* layer = static_cast<InputLayer*>(prev);
    }
    else if(dynamic_cast<FeedForwardLayer*>(prev)){
        FeedForwardLayer* layer = static_cast<FeedForwardLayer*>(prev);
        input = new Matrix(Matrix(layer->get_output(), false).copy_keeping_same_data());
        input->resize(this->error_signal.size(), 1); //use error signal to resize

    }
    else if(dynamic_cast<ConvolutionalLayer*>(prev)){
        ConvolutionalLayer* layer = static_cast<ConvolutionalLayer*>(prev);
        input = new Matrix(Matrix(layer->get_output(), false).copy_keeping_same_data());
        input->resize(weights.width(),1);
    }
    else{
        ilog(FATAL_ERROR, "unrecognized previous layer");
    }
}