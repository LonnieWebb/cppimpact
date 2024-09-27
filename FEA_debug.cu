#include <cblas.h>

#include <chrono>
#include <string>

#include "include/analysis.h"
#include "include/cppimpact_defs.h"
#include "include/elastoplastic.h"
#include "include/mesh.h"
#include "include/physics.h"
#include "include/tetrahedral.h"
#include "include/wall.h"

#ifdef CPPIMPACT_CUDA_BACKEND
#include "include/dynamics.cuh"
#else
#include "include/dynamics.h"
#endif

// Function to print a 3x3 matrix
template <typename T>
void print_matrix(const T M[], const std::string& name) {
  std::cout << name << ":\n";
  for (int i = 0; i < 3; ++i) {
    std::cout << "  ";
    for (int j = 0; j < 3; ++j) {
      std::cout << std::setw(10) << M[i * 3 + j] << " ";
    }
    std::cout << "\n";
  }
}

int main(int argc, char* argv[]) {
  printf("FEA DEBUG\n");
#ifdef CPPIMPACT_CUDA_BACKEND
  using T = double;
#else
  using T = double;
#endif
  using Basis = TetrahedralBasisLinear<T>;
  using Quadrature = TetrahedralQuadrature;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  constexpr int dof_per_node = 3;

  std::vector<std::string> node_set_names;
  // Load in the mesh
  std::string filename("../input/simple tet.inp");
  Mesh<T, Basis::nodes_per_element> tensile;

  // Material Properties
  T E = 68.9E9;  // Pa
  T rho = 2700;  // kg/m3
  T nu = 0.33;
  T beta = 0.0;
  T H = 10;
  T Y0 = 1.9 * std::sqrt(3.0);
  std::string name = "AL6061";

  Elastoplastic<T, dof_per_node> material(E, rho, nu, beta, H, Y0, name);
  tensile.load_mesh(filename);

  const int normal = 1;
  std::string wall_name = "Wall";
  T location = -10;
  Wall<T, 2, Basis> w(wall_name, location, E * 10, tensile.slave_nodes,
                      tensile.num_slave_nodes, normal);

  Dynamics<T, Basis, Analysis, Quadrature> dyna(&tensile, &material, &w);

  // Solve loop with total timer
  auto start = std::chrono::high_resolution_clock::now();
  dyna.debug_strain(0.01, 0, 9.99E-002, 1e-5);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

  return 0;
}