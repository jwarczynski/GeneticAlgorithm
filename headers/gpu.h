#ifndef GPU_H
#define GPU_H

#include "types.h"


#define MAX_THREADS  256

namespace gpu {

	ushort fittest(ushort *chromosome);
	ushort maxDegree();
	// void translate(string name);
	chromosome *generateSmallSample();
	ushort geneticAlg(chromosome *sample, ushort *coloring);
	chromosome *devaluate(chromosome *population, ushort maxColors);
	ushort *mate(ushort *mother, ushort *father, ushort maxColors);
	ushort *minimalizeColors(ushort *chromosome, ushort maxColors);
	ushort colorCount(chromosome *population);
	ushort colorCount(ushort *chromosome);
	void mutate(ushort *chromosome, ushort maxColor, ushort a);
	ushort *crossover(ushort *first, ushort *second);
	chromosome *generatePopulation(ushort maxDegree);
}

#endif
