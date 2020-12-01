#pragma once
#include "TensorTemplate.hpp"
#include "CPUTensor.hpp"
#include "GPUMatrix.hpp"
#include "include/IO.hpp"

class GPUTensor : public AbstractTensor{
public:
    GPUMatrix* m;
    
    GPUTensor(){
        struct_type = StructType::STRUCT_GPUTensor;
        order = 0;
        a = b = 1;
    }

    GPUTensor(size_t X, size_t Y, size_t Z, size_t A = 1, size_t B = 1){
        struct_type = StructType::STRUCT_GPUTensor;
        if(A == 1) B = 1; //prevent abuse
        x = X;
        y = Y;
        z = Z;
        a = A;
        b = B;
        size = x*y*z*a*b;
        m = new GPUMatrix(y*z, x*a*b);
        if(A == 1 && B == 1) order = 3;
        else if(B == 1) order = 4;
        else order = 3;
    }

    GPUTensor(GPUMatrix* M, size_t Z){
        struct_type = StructType::STRUCT_GPUTensor;
        construct_3d(M, Z);
        m = M;
    }

    GPUTensor(GPUMatrix* M, size_t Z, size_t A){
        struct_type = StructType::STRUCT_GPUTensor;
        construct_4d(M, Z, A);
        m = M;
    }

    GPUTensor(GPUMatrix* M, size_t Z, size_t A, size_t B){
        struct_type = StructType::STRUCT_GPUTensor;
        construct_5d(M, Z, A, B);
        m = M;
    }

    

    



    void print() const override{
        CPUTensor* cpu_version = dynamic_cast<CPUTensor*>(copy_to_CPU_tensor());
        cpu_version->print();
        cpu_version->~CPUTensor();
    }

    AbstractMatrix<double>* get_implementation() override{
        return m;
    }

    const AbstractMatrix<double>* get_implementation() const override{
        return m;
    }

    void set_implementation(AbstractMatrix <double>* imp) override{
        m = dynamic_cast<GPUMatrix*>(imp);
    }

     static GPUTensor copy(GPUTensor & x){
        return GPUTensor(x.m->copy(), x.z, x.a , x.b);
    }

    void copy(AbstractTensor* m) const override{
        m = new GPUTensor(static_cast<GPUMatrix*>(m->get_implementation()->copy()), z);
    }

    AbstractTensor* copy_to_CPU_tensor() const override{
        return new CPUTensor(new CPUMatrix(y*z, x, m->copy_to_CPU()), z);
    }
    
    void convolute(AbstractTensor* layer, AbstractTensor* bias, AbstractTensor* out);


};

