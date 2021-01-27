#include "../Training.hpp"
//GPU context switcher

//TODO add output errors to network setup - much like you did for inputs
bool MNIST_conv(){
    Matrix::MaintainState mst;
    Matrix::forceUseCPU();

    TrainCases train_data(0.9);
    train_data.load_dataset_png_byte_training("../mnist_digits");

    Matrix first_test_element(train_data.get_next_case().inputs, false);
    std::vector<uint8_t> image_data(first_test_element.size());
    for(int i = 0; i < image_data.size(); ++i){
        image_data[i] = first_test_element.index(i);
    }

    write_image_to_file(image_data.data(), first_test_element.width(), first_test_element.height(), 8);
    std::cout << "OUTPUT: ";
    train_data.get_current_case().outputs->print();

    // NeuralNetwork NN(NeuralNetwork::ErrorHalfSquared);
    // NN.addLayer(new ConvolutionalLayer(28,28,1,1,3,3,NULL));
    // NN.addLayer(new ConvolutionalLayer(26,26,1,1,26,26,NULL));
    // NN.randomise();

    
    
    NeuralNetwork NN(NeuralNetwork::ErrorHalfSquared);
    NN.addLayer(new ConvolutionalLayer(28,28,1,1,3,3, nullptr));
    NN.addLayer(new ConvolutionalLayer(26,26,1,1,4, 4, nullptr));
    auto x = new FeedForwardLayer(23*23, 1, nullptr);
    NN.addLayer(x);
    NN.randomise();

    Trainer t(&train_data, &NN, 1000);
    
    t.begin_training(0.1);


    TrainCases test_data(0.9);
    test_data.load_dataset_png_byte_testing("../mnist_digits");
    Tester te(&test_data, &NN);
    auto accuracy = te.compute_accuracy();
    std::cout << "Network accuracy: " << accuracy << std::endl;
    if (std::isnan(abs(accuracy))) return false;
    if (accuracy < 0.2) return false;

    return true;
}