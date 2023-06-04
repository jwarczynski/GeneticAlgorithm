#ifndef GPU_H
#define GPU_H

#include "common.h"
#include "types.h"
#include "util.h"
#include <charconv>
#include <sys/types.h>


#define MAX_THREADS  256

namespace gpu {

	int fittest(ushort *chromosome);
	ushort maxDegree();
	void translate(string name);
	vector<pair<vector<int> *, int> *> *generateSmallSample();
	// int geneticAlg(vector<pair<vector<int> *, int> *> *sample, std::vector<int> *coloring);
	ushort geneticAlg(chromosome *sample, ushort *coloring);
	vector<pair<vector<int> *, int> *> *devaluate(vector<pair<vector<int> *, int> *> *population, int maxColors);
	vector<int> *mate(vector<int> *mother, vector<int> *father, int maxColors);
	vector<int> *minimalizeColors(vector<int> *chromosome, int maxColors);
	ushort colorCount(chromosome *population);
	ushort colorCount(ushort *chromosome);
	void mutate(vector<int> *chromosome, int maxColor, int a);
	vector<vector<int> *> *crossover(vector<int> *first, vector<int> *second);
	vector<vector<int> *> *newPop(vector<vector<int> *> *population);
	vector<pair<vector<int> *, int> *> *generatePopulation(int maxDegree);
}

#endif
