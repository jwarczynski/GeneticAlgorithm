#ifndef GPU_H
#define GPU_H

#include "types.h"


#define MAX_THREADS  256

namespace gpu {

	ushort geneticAlg(chromosome *sample, ushort *coloring);
	ushort fittest(ushort *chromosome);
	ushort maxDegree();
	chromosome *generateSmallSample();
	chromosome *devaluate(chromosome *population, ushort maxColors);
	ushort *mate(ushort *mother, ushort *father, ushort maxColors);
	ushort *minimalizeColors(ushort *chromosome, ushort maxColors);
	ushort colorCount(chromosome *population);
	ushort colorCount(ushort *chromosome);
	chromosome *generatePopulation(ushort maxDegree);
	void mutate(ushort *chromosome, ushort maxColor, ushort a);
}

#endif
