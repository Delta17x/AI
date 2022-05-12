// AI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>
#include "AI.h"

int main()
{
    Network network(2, 2, 2, 5);

    float* foo = new float[2];

    std::uniform_real_distribution<float> dist(-200, 200);

    for (int i = 0; i < 1000000; i++) {
        foo[0] = dist(network.GetRandom());
        foo[0] = dist(network.GetRandom());
        float f = network.Predict(foo);
        if (f != 4)
            network.AdjustRandom(0.01f);
    }


    for (auto x : network.Weights()) {
        std::cout << x << "\n";
    }
    foo[0] = 3;
    foo[1] = 2;
    std::cout << network.Predict(foo);

    delete[] foo;


    /*
    * 
    auto t1 = std::chrono::high_resolution_clock::now();
    *
    * 
    auto t2 = std::chrono::high_resolution_clock::now();
    *
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
    */
}

