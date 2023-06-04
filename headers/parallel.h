#ifndef PARALLEL_H
#define PARALLEL_H

#include "types.h"

namespace parallel {

	ushort fittest(const ushort *chromosome);
	ushort maxDegree();
	// void translate(string name);
	chromosome *generateSmallSample();
	ushort geneticAlg(chromosome *sample, ushort *res);
	chromosome *devaluate(chromosome *population, ushort maxColors);
	ushort *mate(ushort *mother, ushort *father, ushort maxColors);
	ushort *minimalizeColors(ushort *chromosome, ushort maxColors);
	ushort colorCount(chromosome *population);
	ushort colorCount(ushort *chromosome);
	chromosome *newPopVol2(chromosome *population, ushort maxColors);
	void mutate(ushort *chromosome, ushort maxColor, ushort a);
	chromosome *generatePopulation(ushort maxDegree);
}

#endif

