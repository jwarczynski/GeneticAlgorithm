#include "../headers/util.h"
#include "../headers/geneticAlgorithm.h"


using namespace std;

namespace seq {
    
    ushort max(ushort a, ushort b) {
        return a >= b ? a : b;
    }
    
	ushort fittest(ushort *chromosome) {
        ushort penalty = 0;
        for (ushort i = 1; i < n; i++) {
            for (ushort j = i + 1; j < n; j++) {
                penalty += (adj[i][j] == 1) && (chromosome[i] == chromosome[j]);
            }
        }
        return penalty;
	}

	ushort maxDegree() {
        ushort maks = 0;
        for (ushort i = 0; i < n; i++) {
            ushort tmp = 0;
            for (ushort j = 0; j < n; j++) {
                tmp += (adj[i][j] == 1);
            }
            maks = max(maks, tmp);
        }
        return maks;
	}

	chromosome *generatePopulation(ushort maxDegree) {
        chromosome *population = (chromosome*)malloc(POPULATION_SIZE * sizeof(chromosome));
        //printf("N: %d\n", n);
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

	ushort colorCount(ushort *chromosome) {
        ushort res = 0;
        for (ushort i=0; i<n; ++i) {
            res = max(res, chromosome[i]);
        }
        return res;
	}

	ushort colorCount(chromosome *population, size_t size) {
			ushort res = 0;
            for (size_t i=0; i<size; ++i) {
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
      //printf("mother ptr: %p\tfather ptr: %p\n", mother, father);
      //printf("mother[37]: %d\n", mother[37]);
        for (ushort i = 0; i < n; i++) {
            ushort a = rand() % 100;
            if (a < 45) {
            // //printf("below 45 a = %d, i = %d\n", a, i);
              child[i] = mother[i];
            } else if (a < 90) {
            // //printf("below 90 a = %d, i = %d\n", a, i);
              child[i] = father[i];
            } else {
            // //printf("above 90 a = %d, i = %d\n", a, i);
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
          ushort mother = rand() % (POPULATION_SIZE / 2);
          ushort father = rand() % (POPULATION_SIZE / 2);
          while (father == mother) {
            father = (father + 1) % (POPULATION_SIZE / 2);
          }
          chromosome child;
          //printf("mate %d chromosome\n", i);
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
            ushort *rescaledIndividual = (ushort*)malloc(n * sizeof(ushort));
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

  int geneticAlg(chromosome *sample, ushort *res) {
      helperFunctionsImpl implementations = {
        seq::maxDegree,
        seq::colorCount,
        seq::generatePopulation,
        seq::newPopVol2,
        seq::minimalizeColors,
        seq::devaluate
      };

      return geneticAlg(sample ,res, implementations);
    }

}
