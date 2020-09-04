#include <iostream>
#include "tests.hpp"
//#include "include/AsyncJob.hpp"
#include "Training.hpp"

class A{
	virtual void a() = 0;
};

int main() {
	
	//XORTest();
	//CNNTest();
	//BigCNNTest();
	//Test::mult();
	//Test::ElementTimed();
	
	/* class A{
		public:
		int a;
		A(int b){
			a = b;
		}
		static int v(A* c, void*){
			std::cout << c->a << std::endl;
			return 0;
		}
	};

	AsyncJob<A> jbs(A::v, 0);

	for(int i = 0; i < 100; i++){
		jbs.addJob(new A(i));
	}	 */

	Test::testAll();
	
	//Test::conv();
	std::string a;
	std::cin >> a;
}


