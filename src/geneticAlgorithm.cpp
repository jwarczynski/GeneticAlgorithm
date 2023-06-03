#include "../headers/geneticAlgorithm.h"


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

