#include "../headers/geneticAlgorithm.h"
#include <cstdlib>
#include <cstring>
#include <sys/types.h>



int chromosomeComparator(const void* a, const void* b) {
    const chromosome* chromosomeA = (const chromosome*)a;
    const chromosome* chromosomeB = (const chromosome*)b;
    
    return chromosomeA->conflicts - chromosomeB->conflicts;
}

void init(ushort *colors, ushort *best, chromosome *population, chromosome *sample, helperFunctionsImpl implementations) {
    *best = implementations.colorCount(sample);
    population = implementations.generatePopulation(*best - 1);
    *colors = implementations.colorCount(population);
  
    for (ushort i=1; i<=SAMPLE_SIZE; ++i) {
        population[POPULATION_SIZE - i] = sample[i-1];
    }
    qsort(population, SAMPLE_SIZE, sizeof(chromosome), chromosomeComparator);
}

void replaceUnusedColors(chromosome *population, ushort colors, ushort * (*replaceUnusedColors)(ushort *, ushort)) {
    for (ushort i=0; i<POPULATION_SIZE; ++i) {
       ushort *tmp = replaceUnusedColors(population[i].genes, colors); 
       population[i].genes = tmp;
    }
} 

void updateResults(chromosome *population, ushort* res, ushort *best, ushort *colors, chromosome * (*limitMaxColor)(chromosome *, ushort)) {
    if (population[0].conflicts == 0) {
          if (colors < best) {
            *best = *colors;
            memcpy(res, population[0].genes, n*sizeof(ushort));
          }
          population = limitMaxColor(population, *best - 1);
          *colors--;
        }
}

void countAndReplaceUnusedColors(chromosome *population, ushort* colors, helperFunctionsImpl implementations) {
    *colors = implementations.colorCount(population);
    replaceUnusedColors(population, *colors, implementations.replaceUnusedColors);
    *colors = implementations.colorCount(population);
}

void nextGeneration(chromosome *population, ushort* res, ushort *best, ushort *colors, helperFunctionsImpl implementations) {
    chromosome *newPopulation = implementations.createNewPopulation(population, *colors);
    population = newPopulation;
    countAndReplaceUnusedColors(population, colors, implementations);
    
    qsort(population, POPULATION_SIZE, sizeof(chromosome), chromosomeComparator);
    updateResults(population, res, best, colors, implementations.devaluate);
}

ushort geneticAlg(chromosome *sample, ushort *res, helperFunctionsImpl implementations) {
    ushort colors;
    ushort best;
    chromosome *population;
    init(&colors, &best, population, sample, implementations);

    // while (since(start).count() < 300000) {
    unsigned int t = 0;
    while (t < iterations) {
        nextGeneration(population, res, &best, &colors, implementations);
        t++;
    }
    return best;
}

