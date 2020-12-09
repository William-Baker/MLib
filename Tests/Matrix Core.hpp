#pragma once
#include "../Training.hpp"
#include <thread>

#include "../CPUTensor.hpp"
#include "../GPUTensor.hpp"

using std::string;


void error_reporter(std::string s, std::string function_name){
    std::cout << "Test \"" << function_name << "\" failed: " << std::endl << "  " << s << std::endl;
}




#define er(msg) error_reporter(msg, (__func__))
#define ts(x) std::to_string(x)
//#define er << (msg) std::stringstream s; er((s << msg).str()) 
namespace MatrixCoreTests{
    bool Test_indeces_cpu(){
        double* arr = new double [6] {1, 2, 3, 4, 5, 6};

        Matrix::forceUseCPU();
        Matrix m(3, 2, arr);
        for(int y = 0; y < 3; y++){
            for(int x = 0; x < 2; x++){
                if(m.index(y, x) != arr[y + 3*x]){
                    er("not eaual at index y: " + ts(y) + " x: " + ts(x));
                    return false;
                }
            }
        }
        return true;
    }

    bool Test_indeces_gpu(){
        double* arr = new double [6]{1, 2, 3, 4, 5, 6};
        Matrix::forceUseGPU();
        Matrix m(3, 2, arr);
        for(int y = 0; y < 3; y++){
            for(int x = 0; x < 2; x++){
                if(m.index(y, x) != arr[y + 3*x]){
                    er("not eaual at index y: " + ts(y) + " x: " + ts(x));
                    return false;
                }
            }
        }
        return true;
    }


    bool Test_copy_cpu()
    {
        Matrix::forceUseCPU();
        Matrix i(4, 3);
        i.randomFill(-100, 100);
        Matrix a = i.copy();
        string s = Matrix::compare(i, a, -100, 100);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }

    bool Test_copy_gpu()
    {
        Matrix::forceUseGPU();
        Matrix i(4, 3);
        i.randomFill(-100, 100);
        
        Matrix a = i.copy();
        
        string s = Matrix::compare(i, a, -100, 100);
        if(s != ""){
            er(s);
            return false;
        } 
        return true;

    }

    bool Test_scale_cpu()
    {
        {
            Matrix::forceUseCPU();
            Matrix A(4, 3);
            Matrix B(4, 3);
            A.randomFill(-10, 10);

            for(int i = 0; i < A.size(); i++){
                B.setIndex(i, A.index(i) * 3);
            }

            Matrix C = A.scale(3);

            string s = Matrix::compare(B, C, -30, 30);

            if(s != ""){
                er(s);
                return false;
            }
        }
        {
            Matrix::forceUseCPU();
            Matrix A(4, 3);
            Matrix B(4, 3);
            A.randomFill(-10, 10);

            for(int i = 0; i < A.size(); i++){
                B.setIndex(i, A.index(i) * 0);
            }

            Matrix C = A.scale(0);

            string s = Matrix::compare(B,C, -0.1, 0.1);

            if(s != ""){
                er(s);
                return false;
            }
        }
        {
            Matrix::forceUseCPU();
            Matrix A(4, 3);
            Matrix B(4, 3);
            A.randomFill(-10, 10);

            for(int i = 0; i < A.size(); i++){
                B.setIndex(i, A.index(i) * 1000);
            }

            Matrix C = A.scale(1000);

            string s = Matrix::compare(B,C, -10000, 10000);

            if(s != ""){
                er(s);
                return false;
            }
        }

        return true;
    }

    bool Test_scale_gpu()
    {
        {
            Matrix::forceUseGPU();
            Matrix A(4, 3);
            Matrix B(4, 3);
            A.randomFill(-10, 10);

            for(int i = 0; i < A.size(); i++){
                B.setIndex(i, A.index(i) * 3);
            }

            Matrix C = A.scale(3);

            string s = Matrix::compare(B,C, -30, 30);

            if(s != ""){
                er(s);
                return false;
            }
        }
        {
            Matrix::forceUseGPU();
            Matrix A(4, 3);
            Matrix B(4, 3);
            A.randomFill(-10, 10);

            for(int i = 0; i < A.size(); i++){
                B.setIndex(i, A.index(i) * 0);
            }

            Matrix C = A.scale(0);

            string s = Matrix::compare(B,C, -0.1, 0.1);

            if(s != ""){
                er(s);
                return false;
            }
        }
        {
            Matrix::forceUseGPU();
            Matrix A(4, 3);
            Matrix B(4, 3);
            A.randomFill(-10, 10);

            for(int i = 0; i < A.size(); i++){
                B.setIndex(i, A.index(i) * 1000);
            }

            Matrix C = A.scale(1000);

            string s = Matrix::compare(B,C, -10000, 10000);

            if(s != ""){
                er(s);
                return false;
            }
        }

        return true;
    }
}

