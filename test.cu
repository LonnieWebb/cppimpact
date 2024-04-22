#include <cblas.h>
#include <cuda_runtime.h>

#include <string>

#include "include/analysis.h"
#include "include/dynamics.h"
#include "include/elastoplastic.h"
#include "include/mesh.h"
#include "include/physics.h"
#include "include/tetrahedral.h"
#include "include/wall.h"

int main(int argc, char *argv[])
{
  using T = double;
  using Basis = TetrahedralBasis<T>;
  using Quadrature = TetrahedralQuadrature;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  const int dof_per_node = 3;

  std::vector<std::string> node_set_names;
  // Load in the mesh
  std::string filename("../input/Dynamics Cube Coarse.inp");
  Mesh<T> tensile;

  // Material Properties
  T E = 2000;  // Pa
  T rho = 7.8; // kg/m3
  T nu = 0.25;
  T beta = 0.0;
  T H = 10;
  T Y0 = 1.9 * std::sqrt(3.0);
  std::string name = "Steel";

  Elastoplastic<T, dof_per_node> material(E, rho, nu, beta, H, Y0, name);
  tensile.load_mesh(filename);

  // Set the number of degrees of freedom

  // Position and velocity in x, y, z
  T init_position[] = {0.1, 0.1, 0.1};
  T init_velocity[] = {0, 0.0, -0.1};

  const int normal = 1;
  std::string wall_name = "Wall";
  T location = 0.0999;
  double dt = 0.00005;
  double time_end = 3;

  Wall<T, 2, Basis> w(wall_name, location, E, tensile.slave_nodes,
                      tensile.num_slave_nodes, normal);

  Dynamics<T, Basis, Analysis> dyna(&tensile, &material, &w);
  dyna.initialize(init_position, init_velocity);

  dyna.solve(dt, time_end);

  // BLAS test
  // T matA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  // T matB[] = {1, 2, 3};
  // T dimA = 3;
  // T dimB = 5;

  // T result[5];
  // memset(result, 0, 5 * sizeof(T));
  // cblas_dgemv(CblasRowMajor, CblasTrans, dimA, dimB, 1.0, matA, dimB, matB,
  // 1,
  //             0.0, result, 1);
  // // Print the values of result
  // for (int i = 0; i < 5; i++) {
  //   printf("result[%d] = %f\n", i, result[i]);
  // }

  // T matA2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  // T matB2[] = {1, 2, 3};
  // T dimA2 = 5;
  // T dimB2 = 3;

  // memset(result, 0, 5 * sizeof(T));
  // cblas_dgemv(CblasRowMajor, CblasNoTrans, dimA2, dimB2, 1.0, matA2, dimB2,
  //             matB2, 1, 0.0, result, 1);
  // // Print the values of result
  // for (int i = 0; i < 5; i++) {
  //   printf("result[%d] = %f\n", i, result[i]);
  // }

  return 0;
}