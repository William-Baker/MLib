#include <IO.hpp>

/* typedef std::ostream& (*manip) (std::ostream&);

class console {

};
//put it in a cpp
template <class T> console& operator<< (console& con, const T& x) {
    #ifdef __VERBOSE__
    std::cout << x;
    #endif
    return con;
}
template <class T> console& operator>>(console& con,  T& x) {
    #ifdef __VERBOSE__
    std::cin >>x;
    #endif
    return con;
}
console& operator<< (console& con, manip manipulator){
    #ifdef __VERBOSE__
    std::cout<<manipulator;
    #endif
    return con;
} */


void IO::IO_log(LOG_TYPE log_type, std::string message, std::string function_name){
   const char* lookup[] = {
      "ERROR       ",
      "WARNING     ",
      "INFO        ",
      "FATAL ERROR "
   };
   fprintf(stderr, "%s", (lookup[log_type] + message + "\n        - " + function_name + "\n").c_str());
   if(log_type == FATAL_ERROR) throw(message);
}


