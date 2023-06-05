#include "../headers/gpu.h"
#include "../headers/geneticAlgorithm.h"
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>


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
      printf("POP_SIZE - SAMPLE_SIZE: %d\n", POPULATION_SIZE -SAMPLE_SIZE);
      for (ushort i = 0; i < (POPULATION_SIZE - SAMPLE_SIZE); i++) {
        ushort *genes = (ushort*)malloc(n*sizeof(ushort));
        for (ushort j = 0; j < n; j++) {
          ushort a = rand() % maxDegree + 1;
          genes[j] = a;
          // if (j == 37) printf("i = %d\tmother[37]: %d\n", i, genes[j]);
        }
        population[i].genes = genes;
        population[i].conflicts = fittest(genes);
      }
      printf("population generated at adress %p\n", population);
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
        printf("COLOR COUNT START\n");
      ushort res = 0;
      for (ushort i=0; i<size; ++i) {
          printf("color %d chromosome\tpopulation[%d].genes[3]:%d\n", i, i, population[i].genes[3]);
          res = max(res, colorCount(population[i].genes));
          printf("colored %d chromosome\n", i);
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
      printf("mother ptr: %p\tfather ptr: %p\n", mother, father);
      printf("mother[37]: %d", mother[37]);
      for (ushort i = 0; i < n; i++) {
        ushort a = rand() % 100;
        if (a < 45) {
            printf("below 45 a = %d, i = %d\n", a, i);
          child[i] = mother[i];
        } else if (a < 90) {
            printf("below 90 a = %d, i = %d\n", a, i);
          child[i] = father[i];
        } else {
          child[i] = -1;
          toMutate[toMutateCounter++] = i;
            printf("above 90 a = %d, i = %d\n", a, i);
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
        printf("created first 10 of new population\n");
        printf("pop ptr: %p\n", population);
        
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
            ushort *rescaledIndividual = (ushort*)malloc(sizeof(ushort));
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
