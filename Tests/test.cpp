#include <iostream>
#include <utility>
#include "Matrix Core.hpp"
#include "Matrix Math.hpp"
#include "Conv.hpp"
#include "FF.hpp"


bool core(){
    return MatrixCoreTests::Test_copyTest_cpu() &&
    MatrixCoreTests::Test_copyTest_gpu() &&
    MatrixCoreTests::Test_indeces_cpu() &&
    MatrixCoreTests::Test_indeces_gpu() &&
    MatrixCoreTests::Test_scale_cpu() &&
    MatrixCoreTests::Test_scale_gpu();
}

bool math(){
    return MatrixMathTests::Test_add() &&
    MatrixMathTests::Test_addAssign() &&
    MatrixMathTests::Test_mult() &&
    MatrixMathTests::Test_multElementMany() &&
    MatrixMathTests::Test_multElementWise() &&
    MatrixMathTests::Test_multMany() &&
    MatrixMathTests::Test_multManySmall() &&
    MatrixMathTests::Test_sigmoid() &&
    MatrixMathTests::Test_subtract() &&
    MatrixMathTests::Test_subtractAssign() &&
    MatrixMathTests::Test_trans() &&
    MatrixMathTests::Test_transMult();
}

bool ff(){
    Test_FF::simpleXOR();
    return true;
}

bool conv(){
    return TensorCoreTests::Test_tensor_copy() &&
    TensorCoreTests::Test_tensor_conv();
}



int main(){
   
    struct Option{
        char c;
        std::string desc;
        decltype(core)* l;
    };
    std::vector<Option> charLambda;
    charLambda.push_back({'c', "core tests", &core});
    charLambda.push_back({'m', "math tests", &math});
    charLambda.push_back({'v', "conv tests", &conv});
    charLambda.push_back({'f', "feed foward", &ff});
    charLambda.push_back({'v', "conv tests", &conv});



    std::cout << "Test options: \n";
    for(auto e : charLambda){
        std::cout << "    " << e.c << " - " << e.desc << std::endl; 
    }

    std::cout << "    a - execute all tests" << std::endl;

    std::string x;
    std::cin >> x;

    if(x.size() == 0) return 1;

    int tests = 0;
    int passed = 0;
    try
    {
        for(auto e : charLambda){
            if((x[0] == 'a') ||  (x[0] == e.c)){
                tests ++;
                std::cout << "\nExecuting test: " << e.desc << std::endl;
                
                if((*(e.l))()){
                    std::cout << e.desc << " test passed\n";
                    passed++;
                }
                else{
                    std::cout << "\"" << e.desc<< "\"" << " test FAILED\n";
                }
            }
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "An erorr occured during testing " << e.what() << '\n';
    }
    
    std::cout << "\n\n   Passed " << passed << " of " << tests << " tests\n";
}