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

// Function to print matrix for manual verification
void print_matrix(const char *name, const double *matrix, int rows, int cols) {
  std::cout << name << ":\n";
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << matrix[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char *argv[]) {
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

  bool smoke_test = false;
  if (argc > 1) {
    if ("-h" == std::string(argv[1]) or "--help" == std::string(argv[1])) {
      std::printf("Usage: ./gpu_test.cu [--smoke]\n");
      exit(0);
    }

    if ("--smoke" == std::string(argv[1])) {
      smoke_test = true;
    }
  }

  std::vector<std::string> node_set_names;
  // Load in the mesh
  std::string filename("../input/0.25 cube calculix linear 5758 elem.inp");
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

  // Set the number of degrees of freedom

  // Position and velocity in x, y, z
  T init_position[] = {0, 0, 0};
  T init_velocity[] = {0, 0.0, -1};

  const int normal = 1;
  std::string wall_name = "Wall";
  T location = -1.501E-1 - 0.00005;
  double dt = 1e-6;
  double time_end = smoke_test ? dt * 100 : 0.5;

  int export_interval = INT_MAX;
#ifdef CPPIMPACT_DEBUG_MODE
  export_interval = 10;
#endif

  Wall<T, 2, Basis> w(wall_name, location, E * 10, tensile.slave_nodes,
                      tensile.num_slave_nodes, normal);

  Dynamics<T, Basis, Analysis, Quadrature> dyna(&tensile, &material, &w);
  dyna.initialize(init_position, init_velocity);

  // Solve loop with total timer
  auto start = std::chrono::high_resolution_clock::now();
  dyna.solve(dt, time_end, export_interval);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

  /*
  // matrix function test
  // Define simple test matrices for B and D
  double B_matrix[6 * 3 * 4];  // B matrix (6 strain components, 3 spatial
                               // dimensions, 4 nodes)
  double D_matrix[6 * 6];      // D matrix (6x6 for 3D isotropic material)
  double B_T_D_B[3 * 4 * 3 *
                 4];  // Result matrix (12x12 for 4 nodes, 3 DOFs per node)

  // Initialize matrices with constant integer values
  memset(B_matrix, 0, sizeof(B_matrix));
  memset(D_matrix, 0, sizeof(D_matrix));
  memset(B_T_D_B, 0, sizeof(B_T_D_B));

  // Manually chosen integers for B_matrix (6 x 3 x 4)
  double B_values[6 * 3 * 4] = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,  // Strain component 1
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,  // Strain component 2
      25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,  // Strain component 3
      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  // Strain component 4
      49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,  // Strain component 5
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72   // Strain component 6
  };
  std::memcpy(B_matrix, B_values, sizeof(B_values));

  // Manually chosen integers for D_matrix (6x6)
  double D_values[6 * 6] = {36, 35, 34, 33, 32, 31,  // D for isotropic material
                                                     // (normal stress coupling)
                            30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                            17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
                            2, 1};
  std::memcpy(D_matrix, D_values, sizeof(D_values));

  // Call the actual Analysis::calculate_B_T_D_B function
  Analysis::calculate_B_T_D_B(B_matrix, D_matrix, B_T_D_B);

  // Print the result matrix B_T_D_B
  print_matrix("B_T_D_B", B_T_D_B, 12,
               12);  // 12x12 matrix for 4 nodes, 3 DOFs per node
  */
  return 0;
}