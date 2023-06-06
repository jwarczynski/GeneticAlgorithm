#ifndef IO_H
#define IO_H

#include "types.h"

#include <string>

void saveColoringToFile(ushort *coloring, const std::string& filename);
void read(std::string name);
void translate(std::string name);

#endif
