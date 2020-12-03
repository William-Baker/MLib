#pragma once
#include "TensorTemplate.hpp"
#include "CPUMatrix.hpp"
#include "include/IO.hpp"


/**
 * Arrange data:
 * Y1X1Z1 Y1X1Z2 Y1X1Z3 Y2X1Z1 Y2X1Z2 Y2X1Z3--Y1X1Z1A2 Y1X1Z2A2 Y1X1Z3A2 Y2X1Z1A2 Y2X1Z2A2 Y2X1Z3A2----Y1X1Z1B2 Y1X1Z2B2 Y1X1Z3B2 Y2X1Z1B2 Y2X1Z2B2 Y2X1Z3B2--Y1X1Z1A2B2 Y1X1Z2A2B2 Y1X1Z3A2B2 Y2X1Z1A2B2 Y2X1Z2A2B2 Y2X1Z3A2B2
 * Y1X2Z1 Y1X2Z2 Y1X2Z3 Y2X2Z1 Y2X2Z2 Y2X2Z3--Y1X2Z1A2 Y1X2Z2A2 Y1X2Z3A2 Y2X2Z1A2 Y2X2Z2A2 Y2X2Z3A2----Y1X2Z1B2 Y1X2Z2B2 Y1X2Z3B2 Y2X2Z1B2 Y2X2Z2B2 Y2X2Z3B2--Y1X2Z1A2B2 Y1X2Z2A2B2 Y1X2Z3A2B2 Y2X2Z1A2B2 Y2X2Z2A2B2 Y2X2Z3A2B2
 * Since CNN compuatations will access all Z layers per X, Y
*/
class CPUTensor : public AbstractTensor{
public:
    CPUMatrix* m;
    
    CPUTensor(){
        struct_type = StructType::STRUCT_CPUTensor;
        order = 0;
        a = b = 1;
    }

    /* CPUTensor(size_t X, size_t Y, size_t Z){
        struct_type = StructType::STRUCT_CPUTensor;
        x = X;
        y = Y;
        z = Z;
        a = b = 1;
        order = 3;
        size = x*y*z;
        m = new CPUMatrix(y*z, x);
        
        //Arrange data:
        //  Y1X1Z1 Y1X1Z2 Y1X1Z3 Y2X1Z1 Y2X1Z2...
        //  Y1X2Z1 Y1X2Z2 Y1X2Z3 Y2X2Z1 Y2X2Z2...
        // Since CNN compuatations will access all Z layers per X, Y
    }

    CPUTensor(size_t X, size_t Y, size_t Z, size_t A){
        struct_type = StructType::STRUCT_CPUTensor;
        x = X;
        y = Y;
        z = Z;
        a = A;
        b = 1;
        order = 4;
        size = x*y*z*a;
        m = new CPUMatrix(y*z, x*a);
        
        //Arrange data:
        //  Y1X1Z1 Y1X1Z2 Y1X1Z3 Y2X1Z1 Y2X1Z2 Y2X1Z3      Y1X1Z1A2 Y1X1Z2A2 Y1X1Z3A2 Y2X1Z1A2 Y2X1Z2A2 Y2X1Z3A2
        //  Y1X2Z1 Y1X2Z2 Y1X2Z3 Y2X2Z1 Y2X2Z2 Y2X2Z3      Y1X2Z1A2 Y1X2Z2A2 Y1X2Z3A2 Y2X2Z1A2 Y2X2Z2A2 Y2X2Z3A2
        // Since CNN compuatations will access all Z layers per X, Y
    } */

/**
 * Arrange data:
 * Y1X1Z1 Y1X1Z2 Y1X1Z3 Y2X1Z1 Y2X1Z2 Y2X1Z3--Y1X1Z1A2 Y1X1Z2A2 Y1X1Z3A2 Y2X1Z1A2 Y2X1Z2A2 Y2X1Z3A2----Y1X1Z1B2 Y1X1Z2B2 Y1X1Z3B2 Y2X1Z1B2 Y2X1Z2B2 Y2X1Z3B2--Y1X1Z1A2B2 Y1X1Z2A2B2 Y1X1Z3A2B2 Y2X1Z1A2B2 Y2X1Z2A2B2 Y2X1Z3A2B2
 * Y1X2Z1 Y1X2Z2 Y1X2Z3 Y2X2Z1 Y2X2Z2 Y2X2Z3--Y1X2Z1A2 Y1X2Z2A2 Y1X2Z3A2 Y2X2Z1A2 Y2X2Z2A2 Y2X2Z3A2----Y1X2Z1B2 Y1X2Z2B2 Y1X2Z3B2 Y2X2Z1B2 Y2X2Z2B2 Y2X2Z3B2--Y1X2Z1A2B2 Y1X2Z2A2B2 Y1X2Z3A2B2 Y2X2Z1A2B2 Y2X2Z2A2B2 Y2X2Z3A2B2
 * Since CNN compuatations will access all Z layers per X, Y
*/
    CPUTensor(size_t X, size_t Y, size_t Z, size_t A = 1, size_t B = 1){
        struct_type = StructType::STRUCT_CPUTensor;
        if(A == 1) B = 1; //prevent abuse
        x = X;
        y = Y;
        z = Z;
        a = A;
        b = B;
        size = x*y*z*a*b;
        m = new CPUMatrix(y*z, x*a*b);
        if(A == 1 && B == 1) order = 3;
        else if(B == 1) order = 4;
        else order = 3;
    }

    CPUTensor(CPUMatrix* M, size_t Z) {
        struct_type = StructType::STRUCT_CPUTensor;
        construct_3d(M, Z);
        m = M;
    }

    CPUTensor(CPUMatrix* M, size_t Z, size_t A){
        struct_type = StructType::STRUCT_CPUTensor;
        construct_4d(M, Z, A);
        m = M;
    }

    CPUTensor(CPUMatrix* M, size_t Z, size_t A, size_t B){ //TODO combine others into this
        struct_type = StructType::STRUCT_CPUTensor;
        construct_5d(M, Z, A, B);
        m = M;
    }

    

    //I/O

    void print() const override{
        for(auto B = 0; B < b; B++){
            for(auto A = 0; A < a; A++){


                for(auto Y = 0; Y < y; Y++){
                    for(auto Z = 0; Z < z; Z++){
                        for(auto X = 0; X < x; X++){
                            std::cout << index(X,Y, Z) << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }

                
                std::cout << "---------------" << std::endl << std::endl;
            }
            std::cout << "=======================" << std::endl << std::endl;
        }
    }


    //Utitlity

    AbstractMatrix<double>* get_implementation() override{
        return m;
    }

    const AbstractMatrix<double>* get_implementation() const override{
        return m;
    }

    void set_implementation(AbstractMatrix<double>* imp) override{
        m = dynamic_cast<CPUMatrix*>(imp);
    }

    double* get_implementation_array() override {return m->get_implementation_array();}
    
    const double* get_implementation_array() const override {return m->get_implementation_array();}

    static CPUTensor copy(CPUTensor & x){
        return CPUTensor(new CPUMatrix(x.m), x.z, x.a , x.b);
    }

    void copy(AbstractTensor* m) const override{
        m = new CPUTensor(new CPUMatrix(m->get_implementation()), z);
    }

    AbstractTensor* copy_to_CPU_tensor() const override{
        return new CPUTensor(new CPUMatrix(m), z);
    }




    //Functionality

    /**
     * @param layer 4D tensor to convolute over the input
     * @param bias n-D tensor treated as an array with size = out.a
     * @param out 3D tensor to store the output of size (input.x-layer.x+1, input.y-layer.y+1, layer.a)
     * 
     */
    void convolute(AbstractTensor* layer, AbstractTensor* bias, AbstractTensor* out){
        for (int oZ = 0; oZ < out->z; oZ++) {
            for (int oX = 0; oX < out->x; oX++) {
                for (int oY = 0; oY < out->y; oY++) {
                    double temp = 0;
                    for (int cX = 0; cX < layer->x; cX++) {
                        for (int cY = 0; cY < layer->y; cY++) {
                            for (int cZ = 0; cZ < layer->z; cZ++) {
                                    //temp += index(oY*inZ + cYZ, oX + cX) * layer->index(cYZ, cX + oZ * convX);
                                    //temp += index(oY + cYZ, oX + cX) * layer->index(cYZ, cX + convX*oZ);
                                    temp += index(oX + cX, oY + cY, cZ) * layer->index(cX, cY, cZ, oZ);
                            }
                        }
                        temp += bias->index(oZ);
                        
                        //out->setIndex(oY * outZ + oZ , oX, tanh(temp));
                        out->setIndex(oX, oY, oZ, tanh(temp));
                    }
                }
            }
        
        }
    }



};



        
