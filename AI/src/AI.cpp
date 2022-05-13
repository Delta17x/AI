// AI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>
#include "AI.h"

int main() {
    constexpr int sampleCount = 1000;
    Network network(2, 4, 4, 4);
    srand(time(0));

    /*
    float* jeff = new float[2];
    jeff[0] = 2;
    jeff[1] = 123;

    for (int i = 0; i < 100; i++) {
        std::cout << network.Predict(jeff) << " ";
        network.AdjustRandom(1);
    }

    delete[] jeff;
    */
    
    std::pair<std::vector<float>, uint32_t>* samples = new std::pair<std::vector<float>, uint32_t>[sampleCount];
    std::uniform_real_distribution<float> dist(-1000, 1000);

    std::vector<float> cur(2);

    for (int i = 0; i < sampleCount; i++) {

        cur[0] = dist(network.GetRandom());
        cur[1] = dist(network.GetRandom());
        samples[i] = std::pair<std::vector<float>, uint32_t>(cur, 1);
    }

    TrainNetwork<std::vector<float>>(network, samples, sampleCount, 100, 100);
    std::cout << network.Predict(cur);
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

