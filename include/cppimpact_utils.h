#pragma once
#include <cstdio>
#include <filesystem>
#include <string>

// Utility function: write an array to a file
template <typename T>
void array_to_txt(std::string txt_path, const T* array, int size) {
  auto abs_path = std::filesystem::absolute(txt_path);
  if (!std::filesystem::is_directory(abs_path.parent_path())) {
    std::fprintf(stderr, "Cannot write to %s, %s does not exist\n",
                 std::string(abs_path).c_str(),
                 std::string(abs_path.parent_path()).c_str());
    return;
  }

  std::FILE* fp = std::fopen(std::string(abs_path).c_str(), "w");

  for (int i = 0; i < size; i++) {
    std::fprintf(fp, "%.16e\n", array[i]);
  }
  std::fclose(fp);
}
