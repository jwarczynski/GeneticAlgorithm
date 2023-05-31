#ifndef GPU_H
#define GPU_H

#include "common.h"
#include "util.h"


#define MAX_THREADS  256

namespace gpu {

	int fittest(const int *chromosome);
	int fittest(vector<int> *chromosome);
	int maxDegree();
	void translate(string name);
	vector<pair<vector<int> *, int> *> *generateSmallSample();
	int geneticAlg(vector<pair<vector<int> *, int> *> *sample, unsigned int iterations);
	vector<pair<vector<int> *, int> *> *devaluate(vector<pair<vector<int> *, int> *> *population, int maxColors);
	vector<int> *mate(vector<int> *mother, vector<int> *father, int maxColors);
	vector<int> *minimalizeColors(vector<int> *chromosome, int maxColors);
	int colorCount(vector<pair<vector<int> *, int> *> *population);
	int colorCount(vector<int> *chromosome);
	void mutate(vector<int> *chromosome, int maxColor, int a);
	vector<vector<int> *> *crossover(vector<int> *first, vector<int> *second);
	vector<vector<int> *> *newPop(vector<vector<int> *> *population);
	vector<pair<vector<int> *, int> *> *generatePopulation(int maxDegree);
}

#endif
