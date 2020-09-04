cmake_minimum_required (VERSION 3.11)

project("NN")

set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")# -ferror-limit=100")# -_CRT_SECURE_NO_WARNINGS")
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")# -ferror-limit=100")# -_CRT_SECURE_NO_WARNINGS")

set(CMAKE_BUILD_TYPE "DEBUG")

enable_language(CUDA)

set(Additional_Sources
    Matrix.cu
    GPUMatrix.cu
    src/IO.cpp
    ##Examples.cpp
    )


set(INCLUDE_DIRS 
    ${PROJECT_SOURCE_DIR}/PNG
    ${PROJECT_SOURCE_DIR}/PNG/libpng-code
    ${PROJECT_SOURCE_DIR}/include
    )
        
add_subdirectory(PNG/libpng-code)

include_directories(${INCLUDE_DIRS})


add_executable(main Source.cpp ${Additional_Sources})

target_link_libraries(main cublas cudart png)