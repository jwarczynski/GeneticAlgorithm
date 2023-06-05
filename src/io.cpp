#include "../headers/io.h"

#include <fstream>
#include <iostream>

void saveColoringToFile(ushort *coloring, const std::string& filename) {
 std::string folder = "Graph-Visualizer/solutions/"; // Określenie folderu docelowego

  // std::ofstream file(filename); // Konkatenacja folderu i nazwy pliku
  std::ofstream file(folder + filename); // Konkatenacja folderu i nazwy pliku

  if (file.is_open()) {
    for (int i = 0; i < n; i++) {
      file << coloring[i] << std::endl;
    }

    file.close();
  } else {
    std::cout << "Nie można otworzyć pliku" << std::endl;
  }
}

