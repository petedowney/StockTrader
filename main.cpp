#include <iostream>
#include <thread>

#include <stdio.h>
#include <fstream>
#include <filesystem>


//#include <conio.h>

using namespace std;

[[noreturn]] void updateNN() {

    while (true) {

        std::cout << "working1" << std::endl;
        std::this_thread::sleep_for(2000ms);
    }
}

[[noreturn]] void predict() {

    while (true) {
        std::cout << "working2" << std::endl;
        std::this_thread::sleep_for(2000ms);
    }
}

int main() {


    std::cout << "Hello, World!" << std::endl;

    std::thread first (updateNN);
    std::thread second (predict);

    first.join();
    second.join();

    return 0;
}
