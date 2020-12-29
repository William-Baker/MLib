#include "../Training.hpp"

bool MNIST_conv(){
    TrainCases train_data(0.9);
    train_data.load_dataset_png_byte("../mnist_digits", true);
    Matrix first_test_element(train_data.get_current_case().inputs, false);
    std::vector<uint8_t> image_data(first_test_element.size());
    for(int i = 0; i < image_data.size(); ++i){
        image_data[i] = first_test_element.index(i);
    }

    for(int i = 0; i < image_data.size(); ++i){
        std::cout << std::to_string(image_data[i]) << " ";
    }
    write_image_to_file(image_data.data(), first_test_element.width(), first_test_element.height(), 8);
    std::cout << "OUTPUT: ";
    train_data.get_current_case().outputs->print();

    NeuralNetwork NN(NeuralNetwork::ErrorHalfSquared);
    NN.addLayer(new ConvLayer(28,28,1,1,3,3,NULL));
    NN.addLayer(new ConvLayer(26,26,1,1,26,26,NULL));
    NN.randomise();

    Trainer t(&train_data, &NN, 1000);
    
    t.beginTraining(0.1);

    return true;
}