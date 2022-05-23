#ifndef NOVA_INCLUDE
#define NOVA_INCLUDE
#include <random>
#include <functional>
#include <vector>

namespace Nova {
	class Network {
		Network();
	public:
		Network(uint32_t inps, uint32_t hLayerCount, uint32_t hNodeCount, uint32_t outs);
		Network& operator= (const Network& other);
		std::mt19937& GetRandom();		
		void AdjustRandom(float maxChange);
		uint32_t InputCount() const;
		uint32_t OutputCount() const;
		uint32_t HiddenLayerCount() const;
		uint32_t HiddenNodeCount() const;
		std::vector<float> Predict(float* inps);
		uint32_t nodeFromLayer(uint32_t layer, uint32_t index);
		uint32_t nodeFromHLayer(uint32_t layer, uint32_t index);
		uint32_t nodesInLayer(uint32_t layer);

		std::vector<float> weights;
		std::vector<float> biases;
		std::vector<float> nodes;
		std::vector<float> nodesNoSig;

	private:
		uint32_t inputCount, outputCount;
		uint32_t hiddenLayerCount, hiddenNodeCount;

		std::mt19937 rand;


	};

	class TrainingData {
	public:
		TrainingData();
		TrainingData(Network& n, const std::vector<float>& c);
		Network& GetNetwork();
		const std::vector<float>& Costs() const;
	private:
		Network network;
		std::vector<float> costs;

	};

	class Trainer {
	public:
		uint32_t inputCount;
		uint32_t outputCount;
		uint32_t hiddenLayerCount;
		uint32_t hiddenLayerNodeCount;
		uint32_t generationCount;

		float learnRate;

		std::pair<float*, size_t>* samples;
		size_t sampleCount;


		Trainer();
		Trainer(uint32_t inpCount, uint32_t outCount, uint32_t hLayerCount, uint32_t hLayerNodeCount, uint32_t generationCount = 1, float lrnRate = 1);
		Trainer(const Network& network, uint32_t generationCount = 1, float lrnRate = 1);

		TrainingData TrainedNetwork();
		Network UntrainedNetwork();

	private:
		float loss(float* out, float* expectedOut);
		float loss(float* out, size_t expectedOut);
		float getAvgCost(Network& network);	
	};

	namespace detail {
		void parallelLoop(uint32_t index1, uint32_t index2, std::function<void(uint32_t start, uint32_t end)> func);

		template<class T>
		struct DynArray {
		public:
			T* ptr;
			inline DynArray(size_t size) {
				ptr = new T[size];
			}
			inline ~DynArray() {
				delete[] ptr;
			}
			inline T& operator[] (size_t index) {
				return *(ptr + index);
			}
		private:
			inline DynArray() {
				ptr = new T[1];
			}
		};

		float sigDeriv(float x);
		float sigmoid(float x);
	}
}

#endif


// TODO: training info class
// TODO: move inline methods to .cpp file
// TODO: custom logger class
// TODO: fix formatting