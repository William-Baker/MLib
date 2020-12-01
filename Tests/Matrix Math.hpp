#pragma once
#include "../Training.hpp"
#include <thread>

#include "../CPUTensor.hpp"
#include "../GPUTensor.hpp"

#include "Matrix Core.hpp"


namespace MatrixMathTests{
    bool Test_trans()
    {
        Matrix::forceUseCPU();
        Matrix CA(4, 3);
        CA.randomFill(-10, 10);
    #ifdef DEBUG
        CA.print();
        std::cout << std::endl;
    #endif
        Matrix CC = CA.transposeNew();
    #ifdef DEBUG
        CC.print();
        std::cout << std::endl;
        Matrix::forceUseGPU();
    #endif
        Matrix A;
        A.copyToThis(CA);
        Matrix C = A.transposeNew();
    #ifdef DEBUG
        C.print();
    #endif
        string s = Matrix::compare(CC, C, -10, 10);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }

    bool Test_transMult()
    {
        Matrix A(3, 5);
        A.randomFill(0, 10);
        Matrix B(5, 4);
        B.randomFill(0, 10);
    #ifdef DEBUG
        A.print();
        std::cout << "\n\n";
        B.print();
        std::cout << "\n\n";
    #endif
        auto A_t = A.transposeNew();
    #ifdef DEBUG
        A_t.print();
        std::cout << "\n\n";
    #endif
        auto B_t = B.transposeNew();
    #ifdef DEBUG
        B_t.print();
        std::cout << "\n\n";
    #endif

        auto normal = A.multiply(B);
        auto amult = A_t.multiplyA(B);
        auto bmult = A.multiplyB(B_t);
        auto abmult = A_t.multiplyAB(B_t);

    #ifdef DEBUG
        normal.print();
        std::cout << "\n\n";
        amult.print();
        std::cout << "\n\n";
        bmult.print();
        std::cout << "\n\n";
        abmult.print();
        std::cout << "\n\n";
    #endif
        string s = Matrix::compare(normal, amult, 0, 300);
        if(s != ""){
            er(s);
            return false;
        }
        s = Matrix::compare(normal, bmult, 0, 300);
        if(s != ""){
            er(s);
            return false;
        }
        s = Matrix::compare(normal, abmult, 0, 300);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }

    bool Test_mult()
    {
    
        Matrix::forceUseCPU();
        Matrix CA(2, 3);
        CA.randomFill(0, 10);
        Matrix CB(3, 2);
        CB.randomFill(0, 10);

        Matrix C = CA.multiply(CB);

        Matrix::forceUseGPU();

        Matrix A, B;
        A.copyToThis(CA);
        B.copyToThis(CB);
        Matrix arr(A.height(), B.width());
        A.multiply(B, arr);
        string s = Matrix::compare(arr, C, 0, 9 * 10);
        if(s != ""){
            er(s);
            return false;
        }
        return true;

    }

    bool Test_multMany()
    {
        Matrix::forceUseCPU();
        Matrix CA(1000, 500);
        CA.randomFill(0, 10);

        Matrix CB(500, 450);
        CB.randomFill(0, 10);


        Timer::start();
        Matrix C = CA.multiply(CB);
        size_t CPU_time = Timer::time();
        #ifdef DEBUG
        std::cout << "CPU Time: " << CPU_time << "us" << std::endl;
        #endif

        Matrix::forceUseGPU();
        Matrix A, B;
        A.copyToThis(CA);
        B.copyToThis(CB);
        const int count = 100;
        std::vector<Matrix> arr;

        for (int i = 0; i < count; i++)
        {
            arr.push_back(Matrix(A.height(), B.width()));
        }


        A.multiply(B, arr[0]); //Warm up
        Timer::start();
        for (int i = 0; i < count; i++)
        {
            A.multiply(B, arr[i]);
            cudaStreamSynchronize(GPUMatrix::stream);
        }

        size_t GPU_average_time = Timer::time() / count;
        #ifdef DEBUG
        std::cout << "Average time: " << GPU_average_time << "us" << std::endl;
        std::cout << "GPU / CPU: " << CPU_time / GPU_average_time << std::endl;
        #endif



        for (int i = 0; i < count; i++)
        {
            string s = Matrix::compare(arr[i], C, 0, 500 * 500 * 10 * 10);
            if(s != ""){
                er(s);
                return false;
            }
        }


        return true;
    }




    bool Test_multManySmall()
    {
        Matrix::forceUseCPU();
        Matrix CA(45, 50);
        CA.randomFill(0, 10);

        Matrix CB(50, 20);
        CB.randomFill(0, 10);


        Timer::start();
        Matrix C = CA.multiply(CB);
        size_t CPU_time = Timer::time();
        #ifdef DEBUG
        std::cout << "CPU Time: " << CPU_time << "us" << std::endl;
        #endif

        Matrix::forceUseGPU();
        Matrix A, B;
        A.copyToThis(CA);
        B.copyToThis(CB);
        const int count = 100;
        std::vector<Matrix> arr;

        for (int i = 0; i < count; i++)
        {
            arr.push_back(Matrix(A.height(), B.width()));
        }


        A.multiply(B, arr[0]); //Warm up
        Timer::start();
        for (int i = 0; i < count; i++)
        {
            A.multiply(B, arr[i]);
            cudaStreamSynchronize(GPUMatrix::stream);
        }

        size_t GPU_average_time = Timer::time() / count;
        #ifdef DEBUG
        std::cout << "Average time: " << GPU_average_time << "us" << std::endl;
        std::cout << "GPU / CPU: " << CPU_time / GPU_average_time << std::endl;
        #endif



        for (int i = 0; i < count; i++)
        {
            string s = Matrix::compare(arr[i], C, 0, 500 * 500 * 10 * 10);
            if(s != ""){
                er(s);
                return false;
            }
        }


        return true;
    }

    bool Test_multElementMany()
    {
        Matrix::forceUseCPU();
        Matrix CA(1000, 500);
        CA.randomFill(0, 10);

        Matrix CB(1000, 500);
        CB.randomFill(0, 10);


        Timer::start();
        Matrix C = CA.multiplyElementWise(CB);
        size_t CPU_time = Timer::time();
        #ifdef DEBUG
        std::cout << "CPU Time: " << CPU_time << "us" << std::endl;
        #endif

        Matrix::forceUseGPU();
        Matrix A, B;
        A.copyToThis(CA);
        B.copyToThis(CB);
        const int count = 100;
        std::vector<Matrix> arr;

        for (int i = 0; i < count; i++)
        {
            arr.push_back(Matrix(A.height(), B.width()));
        }


        A.multiplyElementWise(B, arr[0]); //Warm up
        Timer::start();
        for (int i = 0; i < count; i++)
        {
            A.multiplyElementWise(B, arr[i]);
            cudaStreamSynchronize(GPUMatrix::stream);
        }

        size_t GPU_average_time = Timer::time() / count;
        #ifdef DEBUG
        std::cout << "Average time: " << GPU_average_time << "us" << std::endl;
        std::cout << "GPU / CPU: " << CPU_time / GPU_average_time << std::endl;
        #endif



        for (int i = 0; i < count; i++)
        {
            string s = Matrix::compare(arr[i], C, 0, 500 * 500 * 10 * 10);
            if(s != ""){
                er(s);
                return false;
            }
        }


        return true;
    }



    bool Test_multElementWise()
    {
        Matrix::forceUseCPU();
        Matrix CA(4, 3);
        CA.randomFill(-10, 10);
        Matrix CB(4, 3);
        CB.randomFill(-10, 10);
        Matrix CC(CA.height(), CA.width());
        CA.multiplyElementWise(CB, CC);
        //Matrix CC = CA.multiplyElementWise(CB);
    #ifdef DEBUG
        CA.print();
        std::cout << std::endl;
        CB.print();
        std::cout << std::endl;
        CC.print();
        std::cout << std::endl;
    #endif
        Matrix::forceUseGPU();
        Matrix A;
        A.copyToThis(CA);
        Matrix B;
        B.copyToThis(CB);
        Matrix C(A.height(), A.width());
        A.multiplyElementWise(B, C);
        //Matrix C = A.multiplyElementWise(B);
    #ifdef DEBUG
        C.print();
    #endif
        string s = Matrix::compare(CC, C, -100, 100);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }

    bool Test_sigmoid()
    {
        Matrix::forceUseCPU();
        Matrix CA(4, 3);
        CA.randomFill(-10, 10);
    #ifdef DEBUG
        CA.print();
        std::cout << std::endl;
    #endif
        Matrix CC = CA.sigmoid();
    #ifdef DEBUG
        CC.print();
        std::cout << std::endl;
    #endif
        Matrix::forceUseGPU();

        Matrix A;
        A.copyToThis(CA);
        Matrix C = A.sigmoid();
    #ifdef DEBUG
        C.print();
    #endif
        string s = Matrix::compare(CC, C, 0, 1);
        if(s != ""){
            er(s);
            return false;
        }

        Matrix::forceUseCPU();
    #ifdef DEBUG
        CC.print();
        std::cout << std::endl;
    #endif
        CA = CC.sigmoidDifferential();
    #ifdef DEBUG
        CA.print();
        std::cout << std::endl;
    #endif
        Matrix::forceUseGPU();
        A.copyToThis(CC);
    #ifdef DEBUG
        A.print();
        std::cout << std::endl;
    #endif
        C = A.sigmoidDifferential();
    #ifdef DEBUG
        C.print();
    #endif
        s = Matrix::compare(CA, C, -10, 10);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }

    bool Test_add()
    {
        Matrix::forceUseCPU();
        Matrix CA(4, 3);
        CA.randomFill(-10, 10);
        Matrix CB(4, 3);
        CB.randomFill(-10, 10);
        Matrix CC = CA.add(CB);
    #ifdef DEBUG
        CA.print();
        std::cout << std::endl;
        CB.print();
        std::cout << std::endl;
        CC.print();
        std::cout << std::endl;
    #endif
        Matrix::forceUseGPU();
        Matrix A;
        A.copyToThis(CA);
        Matrix B;
        B.copyToThis(CB);
        Matrix C = A.add(B);
    #ifdef DEBUG
        C.print();
    #endif
        string s = Matrix::compare(CC, C, -20, 20);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }

    bool Test_addAssign()
    {
        Matrix::forceUseCPU();
        Matrix CA(4, 3);
        CA.randomFill(-10, 10);
        Matrix CC;
        CC.copyToThis(CA);
        Matrix CB(4, 3);
        CB.randomFill(-10, 10);
        CA.addAssign(CB);
    #ifdef DEBUG
        CA.print();
        std::cout << std::endl;
    #endif
        Matrix::forceUseGPU();
        Matrix A;
        A.copyToThis(CC);
        Matrix B;
        B.copyToThis(CB);
        A.addAssign(B);
    #ifdef DEBUG
        A.print();
    #endif
        string s = Matrix::compare(CA, A, -20, 20);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }


    bool Test_subtract()
    {
        Matrix::forceUseCPU();
        Matrix CA(4, 3);
        CA.randomFill(-10, 10);
        Matrix CB(4, 3);
        CB.randomFill(-10, 10);
        Matrix CC = CA.subtract(CB);
    #ifdef DEBUG
        CA.print();
        std::cout << std::endl;
        CB.print();
        std::cout << std::endl;
        CC.print();
        std::cout << std::endl;
    #endif
        Matrix::forceUseGPU();
        Matrix A;
        A.copyToThis(CA);
        Matrix B;
        B.copyToThis(CB);
        Matrix C = A.subtract(B);
    #ifdef DEBUG
        C.print();
    #endif
        auto s = Matrix::compare(CC, C, -20, 20);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }

    bool Test_subtractAssign()
    {
        Matrix::forceUseCPU();
        Matrix CA(4, 3);
        CA.randomFill(-10, 10);
        Matrix CC;
        CC.copyToThis(CA);
        Matrix CB(4, 3);
        CB.randomFill(-10, 10);
        CA.subtractAssign(CB);
    #ifdef DEBUG
        CA.print();
        std::cout << std::endl;
    #endif
        Matrix::forceUseGPU();
        Matrix A;
        A.copyToThis(CC);
        Matrix B;
        B.copyToThis(CB);
        A.subtractAssign(B);
    #ifdef DEBUG
        A.print();
    #endif
        auto s = Matrix::compare(CA, A, -20, 20);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }

    bool Test_addConst()
    {
        Matrix::forceUseCPU();
        Matrix CA(4, 3);
        CA.randomFill(-10, 10);
    #ifdef DEBUG
        CA.print();
        std::cout << std::endl;
    #endif
        Matrix CC = CA.addConst(3);
    #ifdef DEBUG
        CC.print();
        std::cout << std::endl;
    #endif
        Matrix::forceUseGPU();

        Matrix A;
        A.copyToThis(CA);
        Matrix C = A.addConst(3);

    #ifdef DEBUG
        C.print();
    #endif
        auto s = Matrix::compare(CC, C, -7, 13);
        if(s != ""){
            er(s);
            return false;
        }
        return true;
    }
}