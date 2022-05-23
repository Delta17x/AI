#include "Nova.h"
#include <iostream>
#include <thread>

using namespace Nova;
uint32_t Network::nodesInLayer(uint32_t layer) {
	if (layer == 0)
		return inputCount;
	if (layer == hiddenLayerCount + 1)
		return outputCount;
	return hiddenNodeCount;
}

uint32_t Network::nodeFromLayer(uint32_t layer, uint32_t index) {
	if (layer == 0)
		return index;
	if (layer == hiddenLayerCount + 1)
		return inputCount + hiddenNodeCount * hiddenLayerCount + index;
	return (layer - 1) * hiddenNodeCount + index + inputCount;
}

uint32_t Network::nodeFromHLayer(uint32_t layer, uint32_t index) {
	return (layer - 1) * hiddenNodeCount + index + inputCount;
}

Network::Network() : inputCount(1), hiddenLayerCount(1), hiddenNodeCount(1), outputCount(1), weights(3), nodes(3), biases(3), nodesNoSig(3), rand(std::random_device()()) {

}

Network::Network(uint32_t inps, uint32_t hLayerCount, uint32_t hNodeCount, uint32_t outs) :
	inputCount(inps), outputCount(outs), hiddenLayerCount(hLayerCount), hiddenNodeCount(hNodeCount), rand(std::random_device()()) {
	size_t weightCount = 0;
	for (uint32_t i = 0; i < hiddenLayerCount + 1; i++) {
		weightCount += nodesInLayer(i) * nodesInLayer(i + 1);
	}
	weights = std::vector<float>(weightCount);
	nodes = std::vector<float>(inps + outs + hiddenLayerCount * hiddenNodeCount);
	biases = std::vector<float>(inps + outs + hiddenLayerCount * hiddenNodeCount);
	nodesNoSig = std::vector<float>(inps + outs + hiddenLayerCount * hiddenNodeCount);

	std::uniform_real_distribution<float> dist(-1, 1);
	for (size_t i = 0; i < biases.size(); i++) {
		biases[i] = dist(rand);
	}
	for (size_t i = 0; i < weights.size(); i++) {
		weights[i] = dist(rand);
	}
}

Network& Network::operator= (const Network& other) {
	inputCount = other.inputCount;
	outputCount = other.outputCount;
	hiddenLayerCount = other.hiddenLayerCount;
	hiddenNodeCount = other.hiddenNodeCount;
	nodes = other.nodes;
	weights = other.weights;
	biases = other.biases;
	rand = other.rand;

	return *this;
}

std::mt19937& Network::GetRandom() {
	return rand;
}

void Network::AdjustRandom(float maxChange) {
	std::uniform_real_distribution<float> dist(-maxChange, maxChange);
	for (size_t i = 0; i < biases.size(); i++) {
		biases[i] += dist(rand);
		weights[i] += dist(rand);
	}
	for (size_t i = biases.size(); i < weights.size(); i++) {
		weights[i] += dist(rand);
	}
}

uint32_t Network::InputCount() const {
	return inputCount;
}

uint32_t Network::OutputCount() const {
	return outputCount;
}

uint32_t Network::HiddenLayerCount() const {
	return hiddenLayerCount;
}

uint32_t Network::HiddenNodeCount() const {
	return hiddenNodeCount;
}

std::vector<float> Network::Predict(float* inps) {
	auto nodesIter = nodes.begin()._Ptr;
	auto nodesNoSIter = nodesNoSig.begin()._Ptr;
	auto weightsIter = weights.begin()._Ptr;

	for (uint32_t i = 0; i < inputCount; i++) {
		*(nodesNoSIter + i) = inps[i];
		*(nodesIter + i) = inps[i];
	}

	uint32_t weightIndex = 0;
	// First hidden layer 


	for (uint32_t j = 0; j < hiddenNodeCount; j++) {
		float dotProd = biases[inputCount + j];
		for (uint32_t k = 0; k < inputCount; k++) {
			dotProd += *(nodesIter + k) * *(weightsIter + weightIndex);
			weightIndex++;
		}

		*(nodesNoSIter + j + inputCount) = dotProd;
		*(nodesIter + j + inputCount) = detail::sigmoid(dotProd);
	}

	// Hidden layers
	for (uint32_t i = 1; i < hiddenLayerCount; i++) {
		for (uint32_t j = 0; j < hiddenNodeCount; j++) {
			float dotProd = biases[inputCount + i * hiddenNodeCount + j];
			for (uint32_t k = 0; k < hiddenNodeCount; k++) {
				dotProd += *(nodesIter + nodeFromHLayer(i, k)) * *(weightsIter + weightIndex);
				weightIndex++;
			}

			uint32_t n = nodeFromHLayer(i + 1, j);

			*(nodesNoSIter + n) = dotProd;
			*(nodesIter + n) = detail::sigmoid(dotProd);
		}
	}

	// output layer
	for (uint32_t j = 0; j < outputCount; j++) {
		float dotProd = biases[inputCount + hiddenLayerCount * hiddenNodeCount + j];
		for (uint32_t k = 0; k < hiddenNodeCount; k++) {
			dotProd += *(nodesIter + nodeFromHLayer(hiddenLayerCount, k)) * *(weightsIter + weightIndex);
			weightIndex++;
		}

		uint32_t n = inputCount + hiddenNodeCount * hiddenLayerCount + j;
		*(nodesNoSIter + n) = dotProd;
		*(nodesIter + n) = detail::sigmoid(dotProd);
	}

	return std::vector<float>(nodes.begin() + inputCount + hiddenNodeCount * hiddenLayerCount, nodes.begin() + inputCount + hiddenNodeCount * hiddenLayerCount + outputCount); // Return outputs
}

Trainer::Trainer() : inputCount(1), outputCount(1), hiddenLayerCount(1), hiddenLayerNodeCount(1), learnRate(1), generationCount(1) {

}

Trainer::Trainer(uint32_t inpCount, uint32_t outCount, uint32_t hLayerCount, uint32_t hLayerNodeCount, uint32_t genCount, float lrnRate) :
	inputCount(inpCount), outputCount(outCount), hiddenLayerCount(hLayerCount),
	hiddenLayerNodeCount(hLayerNodeCount), generationCount(genCount), learnRate(lrnRate) {

}

Trainer::Trainer(const Network& network, uint32_t genCount, float lrnRate) :
	inputCount(network.InputCount()), outputCount(network.OutputCount()), hiddenLayerCount(network.HiddenLayerCount()),
	hiddenLayerNodeCount(network.HiddenNodeCount()), learnRate(lrnRate), generationCount(genCount) {

}

TrainingData Trainer::TrainedNetwork() {
	Network base = Network(inputCount, hiddenLayerCount, hiddenLayerNodeCount, outputCount);
	std::vector<float> costs(generationCount);	
	
	/*
	float avgCost = getAvgCost(base);

	for (uint32_t i = 0; i < generationCount; i++) {
		float cost2 = 0;
		std::vector<float> weightsTemp(base.weights);
		std::vector<float> biasesTemp(base.biases);
		base.AdjustRandom(learnRate);
		cost2 = getAvgCost(base);
		if (cost2 > avgCost) {
			base.weights = weightsTemp;
			base.biases = biasesTemp;
		}
		else
			avgCost = cost2;
		costs[i] = avgCost;
		
	}
	*/

	float* weightsPtr = base.weights.data();
	float* nodesPtr = base.nodes.data();
	float* nodesNoSigPtr = base.nodesNoSig.data();
	float* biasesPtr = base.biases.data();
	
	for (uint32_t currentGen = 0; currentGen < generationCount; currentGen++) {
		for (uint32_t currentSample = 0; currentSample < sampleCount; currentSample++) {
			std::vector<float> gradient(base.weights.size());
			
			base.Predict(samples[currentSample].first);




		}



	}

	TrainingData data(base, costs);
	return data;
}

Network Trainer::UntrainedNetwork() {
	return Network(inputCount, hiddenLayerCount, hiddenLayerNodeCount, outputCount);
}

float Trainer::loss(float* out, float* expectedOut) { // Size of each array is outputCount
	float c = 0;
	for (uint32_t i = 0; i < outputCount; i++) {
		c += (out[i] - expectedOut[i]) * (out[i] - expectedOut[i]);
	}
	return c;
}

float Trainer::loss(float* out, size_t expectedOut) { // Size of each array is outputCount
	float c = 0;
	for (uint32_t i = 0; i < outputCount; i++) {
		if (i != expectedOut)
			c += out[i] * out[i];
		else {
			c += (out[i] - 1) * (out[i] - 1);
		}
	}
	return c;
}

float Trainer::getAvgCost(Network& network) {
	float avgCost = 0;
	for (uint32_t i = 0; i < sampleCount; i++) {
		avgCost += loss(network.Predict(samples[i].first).data(), samples[i].second);
	}
	avgCost /= sampleCount;
	return avgCost;

}

Nova::TrainingData::TrainingData() : network(1, 1, 1, 1), costs(1) {}

Nova::TrainingData::TrainingData(Network& n, const std::vector<float>& c) : costs(c), network(n) {}

Network& Nova::TrainingData::GetNetwork() {
	return network;
}

const std::vector<float>& Nova::TrainingData::Costs() const {
	return costs;
}


void Nova::detail::parallelLoop(uint32_t index1, uint32_t index2, std::function<void(uint32_t start, uint32_t end)> func) {
	using namespace detail;
	uint32_t threadCount = std::thread::hardware_concurrency();
	if (threadCount == 0)
		threadCount = 4;
	std::vector<std::thread> threads(threadCount);

	uint32_t rangeSize = index2 - index1;
	uint32_t size = rangeSize / threadCount;
	uint32_t rem = rangeSize % threadCount;

	for (uint32_t i = 0; i < threadCount; i++) {
		uint32_t a = i * size;
		func(a, a + rangeSize);
	}

	uint32_t a = threadCount * size;
	func(a, a + rem);

	for (uint32_t i = 0; i < threadCount; i++) {
		if (threads[i].joinable())
		threads[i].join();
	}

}

float Nova::detail::sigDeriv(float x) {
	return 1.0f / (2 * (x * x + 2 * fabsf(x) + 1));
}

float Nova::detail::sigmoid(float x) {
	return (x / (1 + fabsf(x)) + 1) / 2.f;
}