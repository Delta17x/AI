#include <iostream>
#include <chrono>
#include <fstream>
#include "Nova.h"

unsigned char** read_mnist_images(const char* full_path, int number_of_images, int image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);
    

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar * [number_of_images];
        for (int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char*)_dataset[i], image_size);
        }
        return _dataset;
    }
    else {
        throw std::runtime_error("Cannot open file");
    }
}

unsigned char* read_mnist_labels(const char* full_path, int number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);
  
    if (file.is_open()) {          
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for (int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    }
    else {
        throw std::runtime_error("Unable to open file");
    }
}

int main() {
    using namespace Nova;
    constexpr int sampleCount = 100;
    constexpr int imgSize = 28 * 28;
    
    std::pair<float*, size_t>* samples = new std::pair<float*, size_t>[sampleCount];
    float** imgs = new float*[sampleCount];
    for (int i = 0; i < sampleCount; i++) {
        imgs[i] = new float[imgSize];
    }
    size_t* labels = new size_t[sampleCount];

    
    unsigned char** imgsU = read_mnist_images("C:\\Users\\Sreekar\\Downloads\\data\\trainimages\\train-images.idx3-ubyte", sampleCount, imgSize);
    unsigned char* labelsU = read_mnist_labels("C:\\Users\\Sreekar\\Downloads\\data\\trainlabels\\train-labels.idx1-ubyte", sampleCount);

    for (int i = 0; i < sampleCount; i++) {
        for (int j = 0; j < imgSize; j++) {
            imgs[i][j] = (float)imgsU[i][j];
        }
        labels[i] = (size_t)labelsU[i];

        samples[i] = std::pair<float*, size_t>(imgs[i], labels[i]);
    }

    Trainer trainer = Trainer(imgSize, 10, 1, 16, 10, 1.0f);
    trainer.samples = samples;
    trainer.sampleCount = sampleCount;

    TrainingData data = trainer.TrainedNetwork();

    auto pred = data.GetNetwork().Predict(samples[43].first); // Prediction for test
    std::cout << "Network's guess: " << std::distance(pred.begin(), std::max_element(pred.begin(), pred.end())) << ". Correct answer: " << samples[43].second;

    delete[] samples, imgs, labels, imgsU, labelsU;

    /*
    * 
    auto t1 = std::chrono::high_resolution_clock::now();


    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";

    */
}

