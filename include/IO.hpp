#pragma once
#include <stdio.h>
#include <string>
#include <iostream>
#include <chrono>
#define __VERBOSE__


template<typename T> void ver(T x){
  #ifdef __VERBOSE__
  std::cout << x;
  #endif
}

template<typename T, typename... Args> void ver(T x, Args... args){
  #ifdef __VERBOSE__
  std::cout << x;
  ver(args...);
  #endif
}



enum LOG_TYPE{
   ERROR=       0,
   WARNING=     1,
   INFO=        2,
   FATAL_ERROR= 3 //Throws an exception
};

namespace IO{
/**
 * Simple macro to log data
 * @param log_type type of data being logged, e.g. ERROR, WARNING, INFO
 * @param message the message you wish to log
 */
void IO_log(LOG_TYPE log_type, std::string message, std::string function_name);
}

#ifndef ilog
   #define ilog(log_type, message) IO::IO_log((log_type), (message), (__func__))
#endif

class Timer {
	static std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
public:
	static void start();
	static size_t time();
};





template<typename T>
static T* allocate_CPU_memory(size_t count){
  auto allocation = std::malloc(count * sizeof(T));
  if(!allocation){
    int attempts = 0;
    while(!allocation){
      allocation = std::malloc(count * sizeof(T));
      std::this_thread::sleep_for(std::chrono::duration<std::chrono::milliseconds>(std::pow(2, attempts)));
      ilog(ERROR, "CPU memory allocation failed");
    }
  }
  return allocation;
}


template<typename T>
static void copy_CPU_memory(T* dst, T* src, size_t count){
  std::memcpy(dst, src, count * sizeof(T));
}





template<typename T>
static T* allocate_GPU_memory(size_t count){
  T* allocation = 0;
  auto err = cudaMalloc(&allocation, count * sizeof(T));
  if(err != cudaSuccess){
    int attempts = 0;
    while(!= cudaSuccess){
      err = cudaMalloc(&allocation, count * sizeof(T));
      std::this_thread::sleep_for(std::chrono::duration<std::chrono::milliseconds>(std::pow(2, attempts)));
      ilog(ERROR, "GPU memory allocation failed" + cudaGetErrorString(err));
    }
  }
  return allocation;
}

template<typename T>
static void copy_GPU_memory(T* dst, T* src, size_t count, cudaMemcpyKind kind){
  auto err = cudaMemcpy(dst, src, size, kind);
  if(err != cudaSuccess){
    int attempts = 0;
    while(!= cudaSuccess){
      err = cudaMemcpy(dst, src, size, kind);
      std::this_thread::sleep_for(std::chrono::duration<std::chrono::milliseconds>(std::pow(2, attempts)));
      ilog(ERROR, "GPU memory copy failed" + cudaGetErrorString(err));
    }
  }
}