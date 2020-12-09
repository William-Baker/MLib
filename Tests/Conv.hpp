#pragma once
#include "../Training.hpp"
#include <thread>

#include "../CPUTensor.hpp"
#include "../GPUTensor.hpp"
#include "Matrix Core.hpp"

namespace TensorCoreTests{
    bool Test_tensor_copy(){
        Matrix::forceUseCPU();
        CPUTensor a(3,3,2);
        a.randomFill(0,10);
        CPUTensor b = CPUTensor::copy(a);

        GPUTensor ga(3,3,2);
        ga.randomFill(0,10);
        GPUTensor gb = GPUTensor::copy(ga);

        Matrix::forceUseGPU();
        Matrix ma(a.get_implementation());
        Matrix mb(b.get_implementation());
        string s = Matrix::compare(ma, mb, -1, 1);
        if(s != ""){
            er(s);
            return false;
        }

        

        Matrix mga(ga.get_implementation());
        Matrix mgb(gb.get_implementation());
        s = Matrix::compare(mga, mgb, -1, 1);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }

    bool Test_tensor_conv_CPU(){

        Matrix::forceUseCPU();

        ConvLayer cv(3, 3, 2, 2, 2, 2,NULL);
        cv.layer.setIndex(0, 0.2);
        cv.layer.setIndex(1, 0.2);
        cv.layer.setIndex(2, 0.4);
        cv.layer.setIndex(3, 0.4);
        cv.layer.setIndex(4, 0.3);
        cv.layer.setIndex(5, 0.3);
        cv.layer.setIndex(6, 0.5);
        cv.layer.setIndex(7, 0.5);
        cv.layer.setIndex(8, 0.2);
        cv.layer.setIndex(9, 0.2);
        cv.layer.setIndex(10, 0.4);
        cv.layer.setIndex(11, 0.4);
        cv.layer.setIndex(12, 0.3);
        cv.layer.setIndex(13, 0.3);
        cv.layer.setIndex(14, 0.5);
        cv.layer.setIndex(15, 0.5);

        cv.bias.setIndex(0, 0);
        cv.bias.setIndex(1, 0);

        Matrix input(6, 3);
        input.setIndex(0, 0.3);
        input.setIndex(1, 0.3);
        input.setIndex(2, 0.2);
        input.setIndex(3, 0.2);
        input.setIndex(4, -0.5);
        input.setIndex(5, -0.5);
        input.setIndex(6, 0.3);
        input.setIndex(7, 0.3);
        input.setIndex(8, 0.9);
        input.setIndex(9, 0.9);
        input.setIndex(10, 0.1);
        input.setIndex(11, 0.1);
        input.setIndex(12, 0.4);
        input.setIndex(13, 0.4);
        input.setIndex(14, 0);
        input.setIndex(15, 0);
        input.setIndex(16, -0.2);
        input.setIndex(17, -0.2);

        cv.compute(input);

        CPUTensor in(static_cast<CPUMatrix*>(input.getStrategy()), 2);
        CPUTensor layer(static_cast<CPUMatrix*>(cv.layer.getStrategy()), 2, 2);
        CPUTensor bias(static_cast<CPUMatrix*>(cv.bias.getStrategy()), 1);
        CPUTensor ot(2,2,2);
        in.convolute(&layer, &bias, &ot);

        Matrix o(ot.get_implementation());
        double* arr = new double[8]{0.876393, 0.876393, 0.309507, 0.309507, 0.793199, 0.793199, 0.235496, 0.235496};
        GPUMatrix b(4, 2, arr, GPUMatrix::MEM::CPU);
        Matrix t(&b);

        //cv.output.print();
        //ot.print();

        string s = Matrix::compare(o,  t, -1, 1);
        if(s != ""){
            er(s);
            return false;
        }
        return true;

        return true;

    }

    bool Test_tensor_conv_GPU(){

        Matrix::forceUseGPU();

        ConvLayer cv(3, 3, 2, 2, 2, 2,NULL);
        cv.layer.setIndex(0, 0.2);
        cv.layer.setIndex(1, 0.2);
        cv.layer.setIndex(2, 0.4);
        cv.layer.setIndex(3, 0.4);
        cv.layer.setIndex(4, 0.3);
        cv.layer.setIndex(5, 0.3);
        cv.layer.setIndex(6, 0.5);
        cv.layer.setIndex(7, 0.5);
        cv.layer.setIndex(8, 0.2);
        cv.layer.setIndex(9, 0.2);
        cv.layer.setIndex(10, 0.4);
        cv.layer.setIndex(11, 0.4);
        cv.layer.setIndex(12, 0.3);
        cv.layer.setIndex(13, 0.3);
        cv.layer.setIndex(14, 0.5);
        cv.layer.setIndex(15, 0.5);

        cv.bias.setIndex(0, 0);
        cv.bias.setIndex(1, 0);

        Matrix input(6, 3);
        input.setIndex(0, 0.3);
        input.setIndex(1, 0.3);
        input.setIndex(2, 0.2);
        input.setIndex(3, 0.2);
        input.setIndex(4, -0.5);
        input.setIndex(5, -0.5);
        input.setIndex(6, 0.3);
        input.setIndex(7, 0.3);
        input.setIndex(8, 0.9);
        input.setIndex(9, 0.9);
        input.setIndex(10, 0.1);
        input.setIndex(11, 0.1);
        input.setIndex(12, 0.4);
        input.setIndex(13, 0.4);
        input.setIndex(14, 0);
        input.setIndex(15, 0);
        input.setIndex(16, -0.2);
        input.setIndex(17, -0.2);

        cv.compute(input);

        CPUTensor in(static_cast<CPUMatrix*>(input.getStrategy()), 2);
        CPUTensor layer(static_cast<CPUMatrix*>(cv.layer.getStrategy()), 2, 2);
        CPUTensor bias(static_cast<CPUMatrix*>(cv.bias.getStrategy()), 1);
        CPUTensor ot(2,2,2);
        in.convolute(&layer, &bias, &ot);

        Matrix o(ot.get_implementation());
        double* arr = new double[8]{0.876393, 0.876393, 0.309507, 0.309507, 0.793199, 0.793199, 0.235496, 0.235496};
        GPUMatrix b(4, 2, arr, GPUMatrix::MEM::CPU);
        Matrix t(&b);

        Matrix::forceUseCPU();
        //cv.output.print();
        //ot.print();

        string s = Matrix::compare(o,  t, -1, 1);
        if(s != ""){
            er(s);
            return false;
        }
        return true;

        return true;

    }
}


