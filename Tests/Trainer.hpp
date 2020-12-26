#include "../Training.hpp"

bool MNIST_conv(){
    TrainCases train_data(0.9);
    train_data.load_dataset_png_byte("../mnist_digits", true);
    train_data.get_current_case().inputs->print();
    std::cout << "OUTPUT: ";
    train_data.get_current_case().outputs->print();

    NeuralNetwork NN(new ErrorHalfSquared);
    NN.addLayer(new ConvLayer(28,28,1,1,3,3,NULL));
    NN.addLayer(new ConvLayer(26,26,1,1,26,26,NULL));
    NN.randomise();

    Trainer t(&train_data, &NN, 1000);

    t.beginTraining(0.1);
    return true;
}