#ifndef TYPES_H
#define TYPES_H

#define POPULATION_SIZE 100
#define SAMPLE_SIZE 3
#define MUTATION_PERCENT 25


#include <vector>
#include <utility>
#include <sys/types.h>


extern ushort n; // number of vertices in graph
extern ushort **adj; // matrix representing graph
extern unsigned int iterations;

					 
struct Node {
    int id;
    int color;
    struct Node *child;
};


typedef struct {
    ushort *genes;
    ushort conflicts;
}chromosome ;


typedef struct {
  ushort (*maxDegree)();
  ushort (*colorCount)(chromosome *);
  chromosome * (*generatePopulation)(ushort);
  chromosome * (*createNewPopulation)(chromosome *, ushort);
  ushort * (*replaceUnusedColors)(ushort *, ushort); 
  chromosome * (*devaluate)(chromosome *, ushort);
} helperFunctionsImpl;


typedef std::vector<std::pair<std::vector<int> *, int> *> population_t;
typedef ushort (*geneticAlgorithm)(chromosome *, ushort *);
// typedef int (*geneticAlgorithm)(population_t *, std::vector<int> *);

#endif
