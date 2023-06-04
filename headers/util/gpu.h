#ifndef GPU_UTIL_H
#define GPU_UTIL_H

// #include "../common.h"
// #include "../util.h"
#include "../types.h"


#define MAX_THREADS  256

namespace gpu {

	int fittest(const int *chromosome);
	int fittest(std::vector<int> *chromosome);
	int maxDegree();
	// void translate(string name);
	std::vector<std::pair<std::vector<int> *, int> *> *generateSmallSample();
	std::vector<std::pair<std::vector<int> *, int> *> *devaluate(std::vector<std::pair<std::vector<int> *, int> *> *population, int maxColors);
	std::vector<int> *mate(std::vector<int> *mother, std::vector<int> *father, int maxColors);
	std::vector<int> *minimalizeColors(std::vector<int> *chromosome, int maxColors);
	int colorCount(std::vector<std::pair<std::vector<int> *, int> *> *population);
	int colorCount(std::vector<int> *chromosome);
	void mutate(std::vector<int> *chromosome, int maxColor, int a);
	std::vector<std::vector<int> *> *crossover(std::vector<int> *first, std::vector<int> *second);
	std::vector<std::vector<int> *> *newPop(std::vector<std::vector<int> *> *population);
	std::vector<std::pair<std::vector<int> *, int> *> *generatePopulation(int maxDegree);
}

#endif
