#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "../headers/types.h"

void benchmarkResults();
void reportResult(geneticAlgorithm implementation, population_t *samplePopulation);

// C
void benchmarkResultsC();
void reportResult(geneticAlgorithmC implementation, chromosome *samplePopulation);


#endif 
