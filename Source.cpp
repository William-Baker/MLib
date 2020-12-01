#include <iostream>
#include "tests.hpp"
#include "Training.hpp"

class A{
	virtual void a() = 0;
};

int main() {
	
	Timer::start();
	size_t x = 0;
	for(size_t i = 0; i < 1000000000; i++){
		x += AbstractTensor::indexing(1,2,3,4,5,i,i/2,i/4, i/5);
	}
	std::cout << "Time 1: " << Timer::time() << " final value: " << x << std::endl;
	x = 0;
	Timer::start();
	for(size_t i = 0; i < 1000000000; i++){
		x += AbstractTensor::indexing_in_one(1,2,3,4,5,i,i/2,i/4, i/5);
	}
	std::cout << "Time 2: " << Timer::time() << " final value: " << x << std::endl;

	//Test::XORTest();
	//Test::CNNTest();
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
	//Test::tensor_copy();
	//Test::testAll();
	//Test::simpleXOR();
	//Test::convBackprop();
	//Test::testAll();
	
	//Test::conv();
	//std::string a;
	//std::cin >> a;
}


