#include <cstdlib> //malloc

#include "../headers/gpu.h"
#include "../headers/geneticAlgorithm.h"



namespace gpu {

    ushort max(ushort a, ushort b) {
        return a >= b ? a : b;
    }
    
    ushort maxDegree() {
      ushort maks = 0;
      for (int i = 0; i < n; i++) {
        ushort tmp = 0;
        for (int j = 0; j < n; j++) {
          if (adj[i][j] == 1) {
            tmp++;
          }
        }
        maks = max(maks, tmp);
      }
      return maks;
    }

    chromosome *generatePopulation(ushort maxDegree) {
      chromosome *population = (chromosome*)malloc(POPULATION_SIZE * sizeof(chromosome));

      for (ushort i = 0; i < (POPULATION_SIZE - SAMPLE_SIZE); i++) {
        ushort *genes = (ushort*)malloc(n*sizeof(ushort));
        for (ushort j = 0; j < n; j++) {
          ushort a = rand() % maxDegree + 1;
          genes[j] = a;
        }
        population[i].genes = genes;
        population[i].conflicts = fittest(genes);
      }

      return population;
    }

    bool isValuePresent(const ushort* arr, size_t size, ushort value) {
      for (size_t i = 0; i < size; i++) {
        if (arr[i] == value) {
          return true;
        }
      }
      return false;
    }

    void mutate(ushort *chromosome, ushort maxColor, ushort a) {
      ushort tabu[n];
      ushort edgesCounter = 0;
      for (ushort i = 0; i < n; ++i) {
        if (adj[a][i] == 1) {
            tabu[edgesCounter++] = chromosome[i];
        }
      }
      int newColor = 1;
      while (isValuePresent(tabu, edgesCounter, newColor)) {
        newColor++;
      }
      if (newColor >= maxColor) {
        newColor = rand() % (maxColor - 1) + 1;
      }
      chromosome[a] = newColor;
    }

    ushort colorCount(chromosome *population, size_t size) {

      ushort res = 0;
      for (ushort i=0; i<size; ++i) {
          res = max(res, colorCount(population[i].genes));
      }
      return res;
    }

    ushort *minimalizeColors(ushort *chromosome, ushort maxColors) {
      ushort *newChromosome = (ushort*)malloc(n*sizeof(ushort));
      ushort colors[n];
      ushort usedColorsCounter[n];
      int lowest = 1;
      
      for (ushort i = 0; i < n; ++i) {
        ++colors[chromosome[i] - 1];
      }
      
      for (ushort i = 0; i < maxColors; i++) {
        if (colors[i] == 0) {
          usedColorsCounter[i] = 1;
        } else {
          usedColorsCounter[i] = ++lowest;
        }
      }
      
      for (int i = 0; i < n; ++i) {
        newChromosome[i] = usedColorsCounter[chromosome[i] - 1];
      }

      return newChromosome;
    }

    ushort *mate(ushort *mother, ushort *father, ushort maxColors) {
      ushort *child = (ushort*)malloc(n*sizeof(ushort));
      ushort toMutate[n];
      ushort toMutateCounter = 0;

      for (ushort i = 0; i < n; i++) {
        ushort a = rand() % 100;
        if (a < 45) {

          child[i] = mother[i];
        } else if (a < 90) {

          child[i] = father[i];
        } else {
          child[i] = -1;
          toMutate[toMutateCounter++] = i;

        }
      }
      for (ushort i=0; i<toMutateCounter; ++i) {
         mutate(child, maxColors, toMutate[i]); 
      }
      return child;
    }

    chromosome *newPopVol2(chromosome *population, ushort maxColors) {
        chromosome *newPopulation = (chromosome*)malloc(POPULATION_SIZE * sizeof(chromosome));
        for (ushort i = 0; i < POPULATION_SIZE / 10; ++i) {
          newPopulation[i] = population[i];
        }
        
        for (ushort i = POPULATION_SIZE / 10; i < POPULATION_SIZE; ++i) {
          ushort mother = rand() % (POPULATION_SIZE / 2) + 1;
          ushort father = rand() % (POPULATION_SIZE / 2) + 1;
          while (father == mother) {
            father = (father + 1) % (POPULATION_SIZE / 2);
          }
          chromosome child;
          child.genes = mate(population[mother].genes, population[father].genes, maxColors);
          child.conflicts = fittest(child.genes);
          newPopulation[i] = child;
        }
        
      return newPopulation;
    }

    chromosome *devaluate(chromosome *population, ushort maxColors) {
        chromosome *newPopulation = (chromosome*)malloc(POPULATION_SIZE * sizeof(chromosome));
         for (ushort i = 0; i < POPULATION_SIZE; ++i) {
            chromosome individual = population[i];
            ushort *rescaledIndividual = (ushort*)malloc(n*sizeof(ushort));
            for (ushort i=0; i< n; ++i) {
                if (individual.genes[i] == maxColors) {
                    rescaledIndividual[i] = individual.genes[i] -1;
                } else {
                    rescaledIndividual[i] = individual.genes[i];
                }
            }
            newPopulation[i].genes = rescaledIndividual;
            newPopulation[i].conflicts = fittest(rescaledIndividual);
      }
      return newPopulation;
    }

    ushort geneticAlg(chromosome *sample, ushort *res) {
      helperFunctionsImpl implementations = {
        gpu::maxDegree,
        gpu::colorCount,
        gpu::generatePopulation,
        gpu::newPopVol2,
        gpu::minimalizeColors,
        gpu::devaluate
      };

      return geneticAlg(sample ,res, implementations);
    }
      
 } 
