#pragma once
#include "Templates.hpp"
#include "include/IO.hpp"


/**
 * Arrange data:
 * Y1X1Z1 Y1X1Z2 Y1X1Z3 Y2X1Z1 Y2X1Z2 Y2X1Z3--Y1X1Z1A2 Y1X1Z2A2 Y1X1Z3A2 Y2X1Z1A2 Y2X1Z2A2 Y2X1Z3A2----Y1X1Z1B2 Y1X1Z2B2 Y1X1Z3B2 Y2X1Z1B2 Y2X1Z2B2 Y2X1Z3B2--Y1X1Z1A2B2 Y1X1Z2A2B2 Y1X1Z3A2B2 Y2X1Z1A2B2 Y2X1Z2A2B2 Y2X1Z3A2B2
 * Y1X2Z1 Y1X2Z2 Y1X2Z3 Y2X2Z1 Y2X2Z2 Y2X2Z3--Y1X2Z1A2 Y1X2Z2A2 Y1X2Z3A2 Y2X2Z1A2 Y2X2Z2A2 Y2X2Z3A2----Y1X2Z1B2 Y1X2Z2B2 Y1X2Z3B2 Y2X2Z1B2 Y2X2Z2B2 Y2X2Z3B2--Y1X2Z1A2B2 Y1X2Z2A2B2 Y1X2Z3A2B2 Y2X2Z1A2B2 Y2X2Z2A2B2 Y2X2Z3A2B2
 * Since CNN compuatations will access all Z layers per X, Y
*/
class AbstractTensor : public MLStruct<double>{
    public:
    size_t x = 0;
    size_t y = 0;
    size_t z = 0;
    size_t a = 1;
    size_t b = 1;

    char order = 0;

    AbstractTensor(){}

    AbstractTensor (AbstractTensor && x){
        set_implementation(x.get_implementation());
        x.set_implementation(nullptr);
    }

    AbstractTensor (AbstractTensor & x) = delete;

    //Virtual functions
    virtual AbstractMatrix<double>* get_implementation() = 0;
    virtual const AbstractMatrix<double>* get_implementation() const  = 0;
    virtual void set_implementation(AbstractMatrix<double>* imp) = 0;

    /* template<typename T>
    T copy(T & a) static{
        return T::copy(a);
    } */
	virtual void copy(AbstractTensor* m) const = 0;
    /**
     * Safe to cast the result to a CPUTensor
     */
    virtual AbstractTensor* copy_to_CPU_tensor() const = 0;

    static inline size_t indexing(size_t X, size_t Y, size_t y_dim){
		return X*y_dim + Y;
	}

    static inline size_t indexing(size_t X, size_t Y, size_t Z, size_t y_dim, size_t z_dim){
		return indexing(Y*z_dim + Z, X,    y_dim);
	}

    static inline size_t indexing(size_t X, size_t Y, size_t Z, size_t A, size_t y_dim, size_t z_dim, size_t a_dim){
		return indexing(A*a_dim + X, Y, Z,    y_dim, z_dim);
	}
	
    static inline size_t indexing(size_t X, size_t Y, size_t Z, size_t A, size_t B, size_t y_dim, size_t z_dim, size_t a_dim, size_t b_dim){
		return indexing(B*b_dim + X, Y, Z, A,   y_dim, z_dim, a_dim);
	}

    static inline size_t indexing_in_one(size_t X, size_t Y, size_t Z, size_t A, size_t B, size_t y_dim, size_t z_dim, size_t a_dim, size_t b_dim){
        return (Y*z_dim + Z) * y_dim + (B*b_dim + (A*a_dim+X));
    }

    //Concrete functions

    /**
     * Returns the i'th element of the implementations array
     * Stored Z Y X major
     */
    double index(size_t i) const {
        return get_implementation()->index(i);
    }

    double index(size_t X, size_t Y) const {
        return get_implementation()->index(Y, X);
    }

    /**
     * Y*z + Z, X
     */
    double index(size_t X, size_t Y, size_t Z) const{
        return get_implementation()->index(Y*z + Z, X);
    }

    /**
     * Y*z + Z, A*a+X
     */
    double index(size_t X, size_t Y, size_t Z, size_t A) const {
        return get_implementation()->index(Y*z + Z, A*a+X);
    }


    /**
     * Y*z + Z, B*b + (A*a+X)
    */
    double index(size_t X, size_t Y, size_t Z, size_t A, size_t B) const{
        return get_implementation()->index(Y*z + Z, B*b + (A*a+X));
    }

    void setIndex(size_t i, double value){
        return get_implementation()->setIndex(i, value);
    }

    void setIndex(size_t X, size_t Y, double value){
        return get_implementation()->setIndex(Y, X, value);
    }

    void setIndex(size_t X, size_t Y, size_t Z, double value){
        return get_implementation()->setIndex(Y*z + Z, X, value);
    }

    void setIndex(size_t X, size_t Y, size_t Z, size_t A, double value){
        return get_implementation()->setIndex(Y*z + Z, A*a+X, value);
    }

    void setIndex(size_t X, size_t Y, size_t Z, size_t A, size_t B, double value){
        return get_implementation()->setIndex(Y*z + Z, B*b + (A*a+X), value);
    }

    void randomFill(double min, double max) {
        get_implementation()->randomFill(min, max);
    }
	void randomFill(double lowerMin, double lowerMax, double upperMin, double upperMax) {
        get_implementation()->randomFill(lowerMin, lowerMax, upperMin, upperMax);
    }

    protected:
    /**
     * handles the validation of construction parameters for all implementations
     */
    void construct_3d(AbstractMatrix<double>* M, size_t Z){
       if(M->y % Z != 0){
            ilog(ERROR, "Matrix Y dimension is not divisible by Z dimension for Tensor");
        }
        x = M->x;
        z = Z;
        y = M->y/z;
        a = b = 1;
        size = x*y*z;
    }

    void construct_4d(AbstractMatrix<double>* M, size_t Z, size_t A){
        if(M->y % Z != 0){
            ilog(ERROR, "Matrix Y dimension is not divisible by Z dimension for Tensor");
        }

        if(M->x % A != 0){
            ilog(ERROR, "Matrix X dimension is not divisible by A dimension for Tensor");
        }
        a = A;
        x = M->x/a;
        z = Z;
        y = M->y/z;
        b = 1;
        size = x*y*z*a;
    }

    void construct_5d(AbstractMatrix<double>* M, size_t Z, size_t A, size_t B){
        if(M->y % Z != 0){
            ilog(ERROR, "Matrix Y dimension is not divisible by Z dimension for Tensor");
        }
        if(M->x % B != 0){
            ilog(ERROR, "Matrix X dimension is not divisible by B dimension for Tensor");
        }
        b = B;
        if(M->x/b % A != 0){
            ilog(ERROR, "Matrix X dimension / B dimension is not divisible by A dimension for Tensor");
        }
        a = A;
        x = M->x/a;
        z = Z;
        y = M->y/z;
        size = x*y*z*a*b;
    }

};