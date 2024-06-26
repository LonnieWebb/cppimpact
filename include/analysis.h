#pragma once

#include <cblas.h>

#include "basematerial.h"
#include "mesh.h"
#include "physics.h"
#include "tetrahedral.h"

template <typename T, class Basis, class Quadrature, class Physics>
class FEAnalysis;
using T = double;
using Basis = TetrahedralBasis<T>;
using Quadrature = TetrahedralQuadrature;
using Physics = NeohookeanPhysics<T>;
// using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

void extract_B_node_block(const T *B, T *B_node, int node, int num_nodes) {
  for (int i = 0; i < 6; ++i) {  // 6 rows for strain components
    for (int j = 0; j < 3;
         ++j) {  // 3 columns for each node's x, y, z displacement derivatives
      B_node[i * 3 + j] = B[i * (3 * num_nodes) + (3 * node + j)];
    }
  }
}

void multiply_BT_node_P(const T *B_node, const T *P, T *BP_node) {
  for (int i = 0; i < 3;
       ++i) {  // Iterate over rows of B^T (and resulting matrix)
    for (int j = 0; j < 3;
         ++j) {  // Iterate over columns of P (and resulting matrix)
      BP_node[i * 3 + j] = 0;  // Initialize the element to 0
      for (int k = 0; k < 6;
           ++k) {  // Iterate over columns of B^T (rows of B_node)
        // Accumulate the product
        BP_node[i * 3 + j] += B_node[k * 3 + i] * P[k * 3 + j];
      }
    }
  }
}

template <typename T, class Basis, class Quadrature, class Physics>
class FEAnalysis {
 public:
  // Static data taken from the element basis
  static constexpr int spatial_dim = 3;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  // Static data from the quadrature
  static constexpr int num_quadrature_pts = Quadrature::num_quadrature_pts;

  // Static data taken from the physics
  static constexpr int dof_per_node = Physics::dof_per_node;

  // Derived static data
  static constexpr int dof_per_element = dof_per_node * nodes_per_element;

  template <int ndof>
  static void get_element_dof(const int nodes[], const T dof[],
                              T element_dof[]) {
    for (int j = 0; j < nodes_per_element; j++) {
      int node = nodes[j];
      for (int k = 0; k < dof_per_node; k++, element_dof++) {
        element_dof[0] = dof[ndof * node + k];
      }
    }
  }

  static void element_mass_matrix(const T element_density,
                                  const T *element_xloc, const T *element_dof,
                                  T *element_mass_matrix_diagonals,
                                  int element_index) {
    for (int i = 0; i < nodes_per_element; i++) {
      T m_i = 0.0;
      for (int k = 0; k < num_quadrature_pts; k++) {
        T pt[spatial_dim];
        T weight = Quadrature::get_quadrature_pt(k, pt);
        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        T J[spatial_dim * spatial_dim];
        Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

        // Compute the inverse and determinant of the Jacobian matrix
        T Jinv[spatial_dim * spatial_dim];
        T detJ = inv3x3(J, Jinv);

        // Compute the invariants
        T N[nodes_per_element];
        Basis::eval_basis_PU(pt, N);
        m_i += N[i] * weight * detJ * element_density;
      }
      element_mass_matrix_diagonals[3 * i] = m_i;
      element_mass_matrix_diagonals[3 * i + 1] = m_i;
      element_mass_matrix_diagonals[3 * i + 2] = m_i;
    }
  }

  static void calculate_f_internal(const T *element_xloc, const T *element_dof,
                                   T *f_internal,
                                   BaseMaterial<T, dof_per_node> *material) {
    T pt[spatial_dim];
    T sigma[6];
    for (int i = 0; i < num_quadrature_pts; i++) {
      T weight = Quadrature::template get_quadrature_pt<T>(i, pt);
      // Evaluate the derivative of the spatial dof in the computational
      // coordinates
      T J[spatial_dim * spatial_dim];
      Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

      // Compute the inverse and determinant of the Jacobian matrix
      T Jinv[spatial_dim * spatial_dim];
      T detJ = inv3x3(J, Jinv);

      T B_matrix[6 * spatial_dim * nodes_per_element];
      memset(B_matrix, 0, 6 * spatial_dim * nodes_per_element * sizeof(T));

      Basis::calculate_B_matrix(Jinv, pt, B_matrix);

      T D_matrix[6 * 6];
      memset(D_matrix, 0, 6 * 6 * sizeof(T));
      Basis::template calculate_D_matrix<dof_per_node>(material, D_matrix);

      T intermediate_1[6];
      memset(intermediate_1, 0, 6 * sizeof(T));

      // multiply B*u
      cblas_dgemv(CblasRowMajor, CblasNoTrans, 6, 30, 1.0, B_matrix, 30,
                  element_dof, 1, 0.0, intermediate_1, 1);

      memset(sigma, 0, 6 * sizeof(T));
      // multiply D*intermediate_1
      cblas_dgemv(CblasRowMajor, CblasNoTrans, 6, 6, 1.0, D_matrix, 6,
                  intermediate_1, 1, 0.0, sigma, 1);

      // T stress_vector[6];
      // stress_vector[0] = sigma[0]; // sigma_xx
      // stress_vector[1] = sigma[4]; // sigma_yy
      // stress_vector[2] = sigma[8]; // sigma_zz
      // stress_vector[3] = sigma[1]; // sigma_xy
      // stress_vector[4] = sigma[5]; // sigma_yz
      // stress_vector[5] = sigma[2]; // sigma_xz

      T BTS[spatial_dim * nodes_per_element];
      memset(BTS, 0, sizeof(T) * spatial_dim * nodes_per_element);

      // multiply B^T by S
      cblas_dgemv(CblasRowMajor, CblasTrans, 6, 30, 1.0, B_matrix, 30, sigma, 1,
                  0.0, BTS, 1);

      for (int j = 0; j < spatial_dim * nodes_per_element; j++) {
        f_internal[j] += weight * detJ * BTS[j];
      }
    }
  }
};

// explicit instantiation if needed

// using T = double;
// using Basis = TetrahedralBasis;
// using Quadrature = TetrahedralQuadrature;
// using Physics = NeohookeanPhysics<T>;
// using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;
// const int nodes_per_element = Basis::nodes_per_element;

// template __global__ void energy_kernel<T, nodes_per_element>(const int
// *element_nodes,
//                                                              const T *xloc,
//                                                              const T *dof, T
//                                                              *total_energy, T
//                                                              *C1, T *D1);

// T J_neo = det3x3(F); // Compute determinant of F to get J
// T lnJ = std::log(detJ);

// T P[spatial_dim * spatial_dim];

// // Initialize P values to 0.0
// for (int j = 0; j < spatial_dim * spatial_dim; ++j)
// {
//   P[j] = 0.0;
// }

// // Compute P: First Piola-Kirchhoff stress tensor
// for (int j = 0; j < 9; ++j)
// {                                // For each component in the 3x3 matrix
//   P[j] = mu * (F[j] - FinvT[j]); // mu * (F - F^{-T})
//   if (j % 4 == 0)
//   {                                  // Diagonal components
//     P[j] += lambda * lnJ * FinvT[j]; // Add lambda * ln(J) * F^{-T} term
//   }
// }

// for (int j = 0; j < 6 * 3 * nodes_per_element; ++j)
// {
//   B_matrix[j] = 0.0;
// }

// T B_node[6 * 3];  // Temporary storage for a 6x3 block of B for a single node
// T BP_node[3 * 3]; // Temporary storage for the result of B_node^T * P for a
// single node

// // Initialize B_node and BP_node to -5.0
// for (int j = 0; j < 18; ++j)
// {
//   B_node[j] = 0.0;
// }
// for (int j = 0; j < 9; ++j)
// {
//   BP_node[j] = 0.0;
// }

// for (int node = 0; node < nodes_per_element; ++node)
// {
//   // Extract the 6x3 block for this node from B
//   extract_B_node_block(B_matrix, B_node, node, nodes_per_element);

//   // Perform the multiplication B_node^T * P
//   cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
//               3, 3, 6,
//               1.0, B_node, 3,
//               P, 3,
//               0.0, BP_node, 3);

//   // Accumulate the result into f_internal for this node
//   for (int j = 0; j < 3; ++j)
//   {
//     for (int k = 0; k < 3; ++k)
//     {
//       f_internal[node * 3 + j] += BP_node[j * 3 + k] * weight * detJ; //
//       Consider the quadrature weight and detJ
//     }
//   }
// }

//---------------------------------------------------------------------

// Compute the deformation gradient F
// T F[spatial_dim * spatial_dim];
// mat3x3MatMult(grad, Jinv, F);
// F[0] += 1.0;
// F[4] += 1.0;
// F[8] += 1.0;

// T mu = material->E / (2 * (1 + material->nu));
// T lambda = material->E * material->nu / ((1 + material->nu) * (1 - 2 *
// material->nu));

// T Finv[spatial_dim * spatial_dim];
// T FinvT[spatial_dim * spatial_dim];
// T FTF[spatial_dim * spatial_dim];
// T FT[spatial_dim * spatial_dim];
// transpose3x3(F, FT);

// mat3x3MatMult(FT, F, FTF);

// T Fdet = inv3x3(F, Finv);
// transpose3x3(Finv, FinvT);

// T sigma[spatial_dim * spatial_dim];

// for (int j = 0; j < 9; j++)
// {
//   sigma[j] = (mu * (F[j] - FinvT[j]) + lambda * (Fdet - 1) * Fdet *
//   FinvT[j]);
// }

// T sigmaFT[spatial_dim * spatial_dim];
// memset(sigmaFT, 0, spatial_dim * spatial_dim * sizeof(T));

// // TODO: double check if output is transposed
// cblas_dgemm(
//     CblasRowMajor, // Specifies that matrices are stored in row-major
//     order, i.e., row elements are contiguous in memory. CblasNoTrans,
//     // Specifies that matrix A (here, 'sigma') will not be transposed
//     before multiplication. CblasTrans,    // Specifies that matrix B
//     (here, 'F') will be transposed before multiplication. spatial_dim,
//     // M: The number of rows in matrices A and C. spatial_dim,   // N:
//     The number of columns in matrices B (after transpose) and C.
//     spatial_dim,   // K: The number of columns in matrix A and rows in
//     matrix B (before transpose), denoting the inner dimension of the
//     product. 1.0 / Fdet,    // alpha: Scalar multiplier applied to the
//     product of matrices A and B. sigma,         // A: Pointer to the
//     first element of matrix A ('sigma') spatial_dim,   // lda: Leading
//     dimension of matrix A. It's the size of the major dimension of A in
//     memory, here equal to 'spatial_dim' because of row-major order. F,
//     // B: Pointer to the first element of matrix B ('F') spatial_dim,
//     // ldb: Leading dimension of matrix B 0.0,           // beta:
//     Scalar multiplier for matrix C before it is updated by the
//     operation. Here, it is set to 0 to ignore the initial content of
//     'sigmaFT'. sigmaFT,       // C: Pointer to the first element of the
//     output matrix C ('sigmaFT'), which will store the result of the
//     operation. spatial_dim    // ldc: Leading dimension of matrix C.
// );