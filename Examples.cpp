/*#include "Matrix.hpp"
#include <stdio.h>
#include <chrono>*/
//
//class Foo
//{
//private:
//	void default_bar(int value)
//	{
//		std::cout << "Nothing entered\n";
//	}
//	std::function<void(Foo*, int)> the_function = &Foo::default_bar;
//	bool Check() {
//		std::string r;
//		std::cin >> r;
//		return std::atoi(r.c_str());
//	}
//public:
//	void replace_bar(std::function<void(Foo*, int)> new_func)
//	{
//		the_function = new_func;
//	}
//
//
//	Foo() {
//		if (Check()) {
// 			the_function = &Foo::baz;
//		}
//		else {
//			the_function = &Foo::default_bar;
//		}
//	}
//
//
//	void bar(int value)
//	{
//		the_function(this, value);
//	}
//
//	void baz(int value)
//	{
//		std::cout << "Something entered\n";
//	}
//};
//
//
//void main() {
//	Foo cla;
//	cla.bar(3);
//}
//

//int main() {
	//Matrix A(2, 2);
	//A.SetIndex(0, 0, 1);
	//A.SetIndex(0, 1, 2);
	//A.SetIndex(1, 0, 3);
	//A.SetIndex(1, 1, 4);

	//Matrix B(2, 2);
	//B.SetIndex(0, 0, 1);
	//B.SetIndex(0, 1, 2);
	//B.SetIndex(1, 0, 3);
	//B.SetIndex(1, 1, 4);

	///*Matrix A(100, 100);
	//Matrix B(100, 100);*/
	//
	//Matrix C = A.Multiply(B);
	//C.print();
		//ret rel no ref											 65249171 
														//					Average loss: 20.255265Total time taken: 63856722
														//ret rel ref		Average loss: 33.466297Total time taken: 81040310
																		//  Average loss: 30.068913Total time taken: 71620142

														// ret deb ref		Average loss: 3.407753Total time taken: 236596657
//}