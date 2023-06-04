#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "types.h"

namespace seq {

	ushort fittest(const ushort *chromosome);
	ushort fittest(ushort *chromosome);
	ushort maxDegree();
	chromosome *generateSmallSample();
	ushort geneticAlg(chromosome *sample, ushort *coloring);
	chromosome *devaluate(chromosome *population, ushort maxColors);
	ushort *mate(ushort *mother, ushort *father, ushort maxColors);
	ushort *minimalizeColors(ushort *chromosome, ushort maxColors);
	ushort colorCount(chromosome *population);
	ushort colorCount(ushort *chromosome);
	void mutate(ushort *chromosome, ushort maxColor, ushort a);
	chromosome *generatePopulation(ushort maxDegree);
}

#endif

