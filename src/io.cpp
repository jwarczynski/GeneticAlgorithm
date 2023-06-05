#include "../headers/io.h"

#include <algorithm>
#include <fstream>
#include <iostream>

void saveColoringToFile(ushort *coloring, const std::string& filename) {
  std::string folder = "Graph-Visualizer/solutions/"; 
  std::ofstream file(folder + filename); 

  if (file.is_open()) {
    ushort maxColor = *std::max_element(coloring, coloring + n);
    file << maxColor << std::endl;
    
    for (int i = 0; i < n; i++) {
      file << coloring[i] << std::endl;
    }

    file.close();
  } else {
    std::cout << "Nie można otworzyć pliku" << std::endl;
  }
}

