#ifndef COMMON_H
#define COMMON_H

#include "util.h"
#define POPULATION_SIZE 100
#define SAMPLE_SIZE 3
#define MUTATION_PERCENT 25

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <algorithm>
#include <vector>
#include <queue>

extern int n; // number of vertices in graph
extern int **adj; // matrix representing graph
extern unsigned int iterations;
					 
struct Node {
    int id;
    int color;
    struct Node *child;
};

typedef struct {
  int (*maxDegree)();
  int (*colorCount)(population_t *);
  vector<pair<vector<int> *, int> *> * (*generatePopulation)(int);
  vector<pair<vector<int> *, int> *> * (*createNewPopulation)(population_t *, int);
  vector<int> * (*minimalizeColors)(vector<int> *, int); 
  vector<pair<vector<int> *, int> *> * (*devaluate)(population_t *, int);
} helperFunctionsImpl;


#endif
