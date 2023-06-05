#include "../headers/geneticAlgorithm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>



int chromosomeComparator(const void* a, const void* b) {
    const chromosome* chromosomeA = (const chromosome*)a;
    const chromosome* chromosomeB = (const chromosome*)b;
    
    return chromosomeA->conflicts - chromosomeB->conflicts;
}

void init(ushort *colors, ushort *best, chromosome **population, chromosome *sample, helperFunctionsImpl implementations) {
    *best = implementations.colorCount(sample, SAMPLE_SIZE);
    printf("first coloring end\n");
    *population = implementations.generatePopulation(*best - 1);
    printf("populatoin generated\n");
    printf("populatoin ptr:%p \n", *population);
    printf("populatoin[0]:%p \n", population[0]->genes);
    printf("populatoin[0][0]:%d \n", population[0]->genes[0]);
    *colors = implementations.colorCount(*population, POPULATION_SIZE - SAMPLE_SIZE);
    printf("second coloring end\n");
  
    for (ushort i=1; i<=SAMPLE_SIZE; ++i) {
        *population[POPULATION_SIZE - i] = sample[i-1];
    }
    qsort(*population, POPULATION_SIZE, sizeof(chromosome), chromosomeComparator);
    printf("populatoin sorted\n");
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
    printf("berofre 3 color count\n");
    *colors = implementations.colorCount(population, POPULATION_SIZE);
    printf("berofre replaceUnusedColors\n");
    replaceUnusedColors(population, *colors, implementations.replaceUnusedColors);
    printf("berofre 4 color count\n");
    *colors = implementations.colorCount(population, POPULATION_SIZE);
}

void nextGeneration(chromosome *population, ushort* res, ushort *best, ushort *colors, helperFunctionsImpl implementations) {
    chromosome *newPopulation = implementations.createNewPopulation(population, *colors);
    printf("created population\n");
    population = newPopulation;
    countAndReplaceUnusedColors(population, colors, implementations);
    
    qsort(population, POPULATION_SIZE, sizeof(chromosome), chromosomeComparator);
    updateResults(population, res, best, colors, implementations.devaluate);
}

ushort geneticAlg(chromosome *sample, ushort *res, helperFunctionsImpl implementations) {
    ushort colors;
    ushort best;
    chromosome *population;
    init(&colors, &best, &population, sample, implementations);

    // while (since(start).count() < 300000) {
    unsigned int t = 0;
    while (t < iterations) {
        nextGeneration(population, res, &best, &colors, implementations);
        t++;
    }
    return best;
}

