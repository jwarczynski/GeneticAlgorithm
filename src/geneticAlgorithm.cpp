#include "../headers/geneticAlgorithm.h"
#include <cstdlib>
#include <cstring>
#include <sys/types.h>


int geneticAlg(population_t *sample, std::vector<int> *res, helperFunctionsImpl implementations) {
    int colors = 0;
    int mDeg;
    if (sample->empty()) {
      mDeg = implementations.maxDegree();
    } else {
      mDeg = implementations.colorCount(sample);
      *res = *sample->at(0)->first;
    }
    vector<pair<vector<int> *, int> *> *population;
    vector<pair<vector<int> *, int> *> *newPopulation;
    population = implementations.generatePopulation(mDeg - 1);
    colors = implementations.colorCount(population);
    for (pair<vector<int> *, int> *s : *sample) {
      population->push_back(s);
    }
    sort(population->begin(), population->end(), comp);
    unsigned int t = 0;
    int best = mDeg;

    // while (since(start).count() < 300000) {
    while (t < iterations) {
        t++;
        newPopulation = implementations.createNewPopulation(population, colors);
        population = newPopulation;
        colors = implementations.colorCount(population);
        for (auto &i : *population) {
          vector<int> *tmp = implementations.minimalizeColors(i->first, colors);
          i->first = tmp;
        }

        colors = implementations.colorCount(population);
        sort(population->begin(), population->end(), comp);
        if (population->at(0)->second == 0) {
          if (colors < best) {
            best = colors;
            *res = *population->at(0)->first;
          }
          population = implementations.devaluate(population, best - 1);
          colors--;
        }
    }
    return best;
}


int chromosomeComparator(const void* a, const void* b) {
    const chromosome* chromosomeA = (const chromosome*)a;
    const chromosome* chromosomeB = (const chromosome*)b;
    
    return chromosomeA->conflicts - chromosomeB->conflicts;
}

void init(ushort *colors, ushort *best, chromosome *population, chromosome *sample, helperFunctionsImplC implementations) {
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

void countAndReplaceUnusedColors(chromosome *population, ushort* colors, helperFunctionsImplC implementations) {
    *colors = implementations.colorCount(population);
    replaceUnusedColors(population, *colors, implementations.replaceUnusedColors);
    *colors = implementations.colorCount(population);
}

void nextGeneration(chromosome *population, ushort* res, ushort *best, ushort *colors, helperFunctionsImplC implementations) {
    chromosome *newPopulation = implementations.createNewPopulation(population, *colors);
    population = newPopulation;
    countAndReplaceUnusedColors(population, colors, implementations);
    
    qsort(population, POPULATION_SIZE, sizeof(chromosome), chromosomeComparator);
    updateResults(population, res, best, colors, implementations.devaluate);
}

ushort geneticAlg(chromosome *sample, ushort *res, helperFunctionsImplC implementations) {
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

