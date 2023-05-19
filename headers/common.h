#ifndef COMMON_H
#define COMMON_H

#define POPULATION_SIZE 100
#define SAMPLE_SIZE 3
#define MUTATION_PERCENT 25

#include <cstdlib>
#include <iostream>
#include <random>
#include <ctime>
#include <algorithm>
#include <vector>

extern int n; // number of vertices in graph
extern int **adj; // matrix representing graph
					 
struct Node {
    int id;
    int color;
    struct Node *child;
};

#endif
