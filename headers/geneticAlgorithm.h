#include "common.h"
#include "types.h"
#include "util.h"
#include <sys/types.h>


int geneticAlg(population_t *sample, std::vector<int> *res, helperFunctionsImpl implementations);
ushort geneticAlg(chromosome *sample, ushort *res, helperFunctionsImplC implementations);
