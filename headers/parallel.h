#ifndef PARALLEL_H
#define PARALLEL_H

#include "types.h"

namespace parallel {

	ushort geneticAlg(chromosome *sample, ushort *res);
	ushort fittest(const ushort *chromosome);
	ushort maxDegree();
	chromosome *generateSmallSample();
	chromosome *devaluate(chromosome *population, ushort maxColors);
	ushort *mate(ushort *mother, ushort *father, ushort maxColors);
	ushort *minimalizeColors(ushort *chromosome, ushort maxColors);
	ushort colorCount(chromosome *population, size_t size);
	ushort colorCount(ushort *chromosome);
	chromosome *newPopVol2(chromosome *population, ushort maxColors);
	chromosome *generatePopulation(ushort maxDegree);
	void mutate(ushort *chromosome, ushort maxColor, ushort a);
}

#endif

