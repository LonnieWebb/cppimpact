#include <cblas.h>

#include <chrono>
#include <cstdlib>
#include <string>

#include "include/analysis.h"
#include "include/cppimpact_defs.h"
#include "include/elastoplastic.h"
#include "include/physics.h"
#include "include/simulation_config.h"
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

  // Default values
  int input_file = 0;
  int deformation = 0;

  // Parse command-line arguments
  if (argc >= 2) {
    input_file = std::atoi(argv[1]);
  }
  if (argc >= 3) {
    deformation = std::atoi(argv[2]);
  }

#ifdef CPPIMPACT_CUDA_BACKEND
  using T = double;
#else
  using T = double;
#endif

#if defined(USE_LINEAR_BASIS)
  using Basis = TetrahedralBasisLinear<T>;
#elif defined(USE_QUADRATIC_BASIS)
  using Basis = TetrahedralBasisQuadratic<T>;
#else
#error "No basis type defined"
#endif

  using Quadrature = TetrahedralQuadrature5pts;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  constexpr int dof_per_node = 3;

  std::vector<std::string> node_set_names;
  // Load in the mesh
  // std::string filename("../input/quad tet.inp");
  // std::string filename("../input/0.25 cube calculix quad 5758 elem.inp");
  Mesh<T, Basis::nodes_per_element> tensile;

  std::string filename;
#if defined(USE_LINEAR_BASIS)
  switch (input_file) {
    case 0:
      filename = "../input/linear tet.inp";
      printf("Using linear tetrahedral element\n");
      break;
    case 1:
      filename = "../input/0.25 cube calculix linear 5758 elem.inp";
      printf("Using linear tetrahedral cube\n");

      break;
    default:
      std::cerr << "Invalid input_file for linear basis: " << input_file
                << std::endl;
      return 1;
  }

#elif defined(USE_QUADRATIC_BASIS)
  switch (input_file) {
    case 0:
      filename = "../input/quad tet.inp";
      printf("Using quadratic tetrahedral element\n");

      break;
    case 1:
      filename = "../input/0.25 cube calculix quad 39247 elem.inp";
      printf("Using quadratic tetrahedral cube\n");

      break;
    default:
      std::cerr << "Invalid input_file for linear basis: " << input_file
                << std::endl;
      return 1;
  }
#else
#error "No basis type defined"
#endif

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
  // T init_position[] = {-9.99E-2, -9.99E-2, 1.501E-1};
  // // T init_position[] = {0, 0, 0};
  // T init_velocity[] = {0, 0.0, -1};
  // dyna.initialize(init_position, init_velocity);

  // Solve loop with total timer
  auto start = std::chrono::high_resolution_clock::now();
  dyna.debug_strain(0.01, deformation);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

  return 0;
}