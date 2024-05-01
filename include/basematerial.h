#pragma once

#include <string>

template <typename T, int dof_per_node>
class BaseMaterial {
 public:
  int ndof;
  std::string name;
  std::string type;
  T E;
  T rho;
  T nu;

  BaseMaterial(std::string type, T E, T rho, T nu,
               std::string name = "Material")
      : name(name), type(type), E(E), rho(rho), nu(nu) {}

  virtual ~BaseMaterial() {}
};
