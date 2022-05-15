#pragma once
#include <random>
#include <vector>

class Network {
	uint32_t inputCount, outputCount;
	uint32_t hiddenLayerCount, hiddenNodeCount;
	std::vector<float> nodes;
	std::vector<float> weights;
	std::vector<float> biases;
	// TODO: make this static and add a .cpp file for its definition (and definition of other inline functions)
	std::mt19937 rand;


	inline size_t nodesInLayer(uint32_t layer) {
		if (layer == 0)
			return inputCount;
		if (layer == hiddenLayerCount + 1)
			return outputCount;
		return hiddenNodeCount;
	}

	inline float& nodeFromLayer(uint32_t layer, uint32_t index) {
		if (layer == 0)
			return nodes[index];
		if (layer == hiddenLayerCount + 1)
			return nodes[inputCount + hiddenNodeCount * hiddenLayerCount + index];
		return nodes[(layer - 1) * hiddenNodeCount + index + inputCount];
	}

	inline float& nodeFromHLayer(uint32_t layer, uint32_t index) {
		return nodes[(layer - 1) * hiddenNodeCount + index + inputCount];
	}

	inline float sizeOfLayer(uint32_t layer) {
		if (layer == 0)
			return inputCount;
		if (layer == hiddenLayerCount + 1)
			return outputCount;
		return hiddenNodeCount;
	}

	inline float sigmoid(float x) {
		return (x / (1 + fabsf(x)) + 1) / 2.f;
	}

	template<class ContType>
	inline float cost(ContType out, ContType expectedOut) { // Size of each array is outputCount
		float c = 0;
		for (int i = 0; i < outputCount; i++) {
			c += (out[i] - expectedOut[i]) * (out[i] - expectedOut[i]);
		}
		return c;
	}
public:
	inline Network() : inputCount(1), hiddenLayerCount(1), hiddenNodeCount(1), outputCount(1), weights(3), nodes(3), biases(3), rand(std::random_device()()) {

	}

	inline Network(uint32_t inps, uint32_t hLayerCount, uint32_t hNodeCount, uint32_t outs) :
		inputCount(inps), outputCount(outs), hiddenLayerCount(hLayerCount), hiddenNodeCount(hNodeCount), rand(std::random_device()()) {
		size_t weightCount = 0;
		for (uint32_t i = 0; i < hiddenLayerCount + 1; i++) {
			weightCount += nodesInLayer(i) * nodesInLayer(i + 1);
		}
		weights = std::vector<float>(weightCount);
		nodes = std::vector<float>(inps + outs + hiddenLayerCount * hiddenNodeCount);
		biases = std::vector<float>(inps + outs + hiddenLayerCount * hiddenNodeCount);

		std::uniform_real_distribution<float> dist(-1, 1);
		for (int i = 0; i < biases.size(); i++) {
			biases[i] = dist(rand);
		}
		for (int i = 0; i < weights.size(); i++) {
			weights[i] = dist(rand);
		}
	}

	inline Network& operator= (const Network& other) {
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

	inline std::mt19937& GetRandom() {
		return rand;
	}

	inline std::vector<float>& Weights() {
		return weights;
	}

	inline const std::vector<float>& Weights() const {
		return weights;
	}

	inline void AdjustRandom(float maxChange) {
		std::uniform_real_distribution<float> dist(-maxChange, maxChange);
		for (int i = 0; i < biases.size(); i++) {
			biases[i] += dist(rand);
			weights[i] += dist(rand);
		}
		for (int i = biases.size(); i < weights.size(); i++) {
			weights[i] += dist(rand);
		}
	}

	template<class InpArrayType>
	inline std::vector<float> Predict(InpArrayType inps) {

		for (uint32_t i = 0; i < inputCount; i++) {
			nodes[i] = inps[i];
		}

		uint32_t weightIndex = 0;
		// First hidden layer 
		float* lastLayerNodes = new float[inputCount];
		float* weightsToLast = new float[inputCount];

		for (uint32_t j = 0; j < hiddenNodeCount; j++) {
			for (uint32_t k = 0; k < inputCount; k++) {
				lastLayerNodes[k] = nodes[k];
				weightsToLast[k] = weights[weightIndex++];

			}

			float dotProd = biases[inputCount + j];
			for (uint32_t k = 0; k < inputCount; k++) {
				dotProd += lastLayerNodes[k] * weightsToLast[k];
			}

			nodes[j + inputCount] = sigmoid(dotProd);
		}

		delete[] lastLayerNodes, weightsToLast;
		lastLayerNodes = new float[hiddenNodeCount];
		weightsToLast = new float[hiddenNodeCount];

		// Hidden layers
		for (uint32_t i = 1; i < hiddenLayerCount; i++) {
			for (uint32_t j = 0; j < hiddenNodeCount; j++) {
				for (uint32_t k = 0; k < hiddenNodeCount; k++) {
					lastLayerNodes[k] = nodeFromHLayer(i, k);
					weightsToLast[k] = weights[weightIndex++];
				}

				float dotProd = biases[inputCount + i * hiddenNodeCount + j];
				for (uint32_t k = 0; k < hiddenNodeCount; k++) {
					dotProd += lastLayerNodes[k] * weightsToLast[k];
				}

				nodeFromHLayer(i + 1, j) = sigmoid(dotProd);
			}
		}

		// output layer
		for (uint32_t j = 0; j < outputCount; j++) {
			for (uint32_t k = 0; k < hiddenNodeCount; k++) {
				lastLayerNodes[k] = nodeFromHLayer(hiddenLayerCount, k);
				weightsToLast[k] = weights[weightIndex++];
			}

			float dotProd = biases[inputCount + hiddenLayerCount * hiddenNodeCount + j];
			for (uint32_t k = 0; k < hiddenNodeCount; k++) {
				dotProd += lastLayerNodes[k] * weightsToLast[k];
			}

			nodeFromLayer(hiddenLayerCount + 1, j) = sigmoid(dotProd);
		}

		delete[] lastLayerNodes, weightsToLast;

		return std::vector<float>(nodes.begin() + inputCount + hiddenNodeCount * hiddenLayerCount, nodes.begin() + inputCount + hiddenNodeCount * hiddenLayerCount + outputCount);
	}

	template<class InpArrayType>
	inline void TrainNetwork(std::pair<InpArrayType, uint32_t>* samples, size_t sampleCount, size_t generationCount) {

	}
};

