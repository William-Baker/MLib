#pragma once
#include <stdio.h>
#include <string>
#include <iostream>
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

