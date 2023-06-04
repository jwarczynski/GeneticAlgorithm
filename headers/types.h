#ifndef TYPES_H
#define TYPES_H

#define POPULATION_SIZE 100
#define SAMPLE_SIZE 3
#define MUTATION_PERCENT 25


#include <vector>
#include <utility>


extern int n; // number of vertices in graph
extern int **adj; // matrix representing graph
extern unsigned int iterations;


typedef std::vector<std::pair<std::vector<int> *, int> *> population_t;
typedef int (*geneticAlgorithm)(population_t *, std::vector<int> *);
					 
struct Node {
    int id;
    int color;
    struct Node *child;
};

typedef struct {
  int (*maxDegree)();
  int (*colorCount)(population_t *);
  std::vector<std::pair<std::vector<int> *, int> *> * (*generatePopulation)(int);
  std::vector<std::pair<std::vector<int> *, int> *> * (*createNewPopulation)(population_t *, int);
  std::vector<int> * (*minimalizeColors)(std::vector<int> *, int); 
  std::vector<std::pair<std::vector<int> *, int> *> * (*devaluate)(population_t *, int);
} helperFunctionsImpl;

typedef struct {
    unsigned short *genes;
    unsigned short  conflicts;
}chromosome ;


#endif
