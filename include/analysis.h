#pragma once

#include <cblas.h>

#include "basematerial.h"
#include "cppimpact_blas.h"
#include "cppimpact_defs.h"
#include "mesh.h"
#include "physics.h"
#include "simulation_config.h"
#include "tetrahedral.h"

template <typename T, class Basis, class Quadrature, class Physics>
class FEAnalysis;

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
  static CPPIMPACT_FUNCTION void get_element_dof(const int nodes[],
                                                 const T dof[],
                                                 T element_dof[]) {
    for (int j = 0; j < nodes_per_element; j++) {
      int node = nodes[j];
      for (int k = 0; k < dof_per_node; k++, element_dof++) {
        element_dof[0] = dof[ndof * node + k];
      }
    }
  }

#ifdef CPPIMPACT_CUDA_BACKEND
  template <int ndof, int dof_per_element, int dof_per_node>
  static __device__ void get_element_dof(int tid, const int nodes[],
                                         const T dof[], T element_dof[]) {
    if (tid < dof_per_element) {
      int j = tid / dof_per_node;
      int k = tid % dof_per_node;
      int node = nodes[j];

      element_dof[tid] = dof[ndof * node + k];
    }
  }
#endif

#ifdef CPPIMPACT_CUDA_BACKEND
  static __device__ void element_mass_matrix_gpu(
      int tid, const T element_density, const T *element_xloc,
      const T *element_dof, T *element_mass_matrix_diagonals,
      const int nodes_per_elem_num_quad) {
    int i = tid / num_quadrature_pts;  // node index
    int k = tid % num_quadrature_pts;  // quadrature index

    __shared__ T m_i[nodes_per_element];
    if (tid < nodes_per_element) {
      m_i[tid] = 0.0;
    }

    __shared__ T pts[num_quadrature_pts * spatial_dim];
    __shared__ T coeff[num_quadrature_pts];
    __shared__ T J[num_quadrature_pts * spatial_dim * spatial_dim];

    int pts_offset = k * spatial_dim;
    int J_offset = k * spatial_dim * spatial_dim;

    __syncthreads();

    // Compute density * weight * detJ for each quadrature point
    if (tid < num_quadrature_pts) {
      coeff[k] = Quadrature::get_quadrature_pt(k, pts + pts_offset);
    }
    __syncthreads();

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates
    Basis::template eval_grad<spatial_dim>(pts + pts_offset, element_xloc,
                                           J + J_offset);
    __syncthreads();

    if (tid < num_quadrature_pts) {
      coeff[k] *= det3x3(J + J_offset) * element_density;
    }
    __syncthreads();

    if (tid < nodes_per_elem_num_quad) {
      // Compute the invariants
      T N[nodes_per_element];
      Basis::eval_basis(pts + pts_offset, N);
      atomicAdd(&m_i[i], N[i] * coeff[k]);
    }

    __syncthreads();

    if (i < nodes_per_element && k < 3)
      element_mass_matrix_diagonals[3 * i + k] = m_i[i];
    __syncthreads();
  }
#endif

  static void element_mass_matrix(const T element_density,
                                  const T *element_xloc, const T *element_dof,
                                  T *element_mass_matrix_diagonals) {
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
        Basis::eval_basis(pt, N);
        m_i += N[i] * weight * detJ * element_density;
      }
      element_mass_matrix_diagonals[3 * i] = m_i;
      element_mass_matrix_diagonals[3 * i + 1] = m_i;
      element_mass_matrix_diagonals[3 * i + 2] = m_i;
    }
  }

  static __device__ void calculate_f_internal_gpu(
      int tid, const T *element_xloc, const T *element_dof, T *f_internal,
      BaseMaterial<T, dof_per_node> *material) {
    T sigma[6];
    int constexpr dof_per_element = spatial_dim * nodes_per_element;
    int i = tid / num_quadrature_pts;  // node index
    int k = tid % num_quadrature_pts;  // quadrature index

    // contiguous for each quadrature points
    __shared__ T pts[num_quadrature_pts * spatial_dim];
    __shared__ T dets[num_quadrature_pts];
    __shared__ T wts[num_quadrature_pts];
    __shared__ T J[num_quadrature_pts * spatial_dim * spatial_dim];
    __shared__ T Jinv[num_quadrature_pts * spatial_dim * spatial_dim];
    __shared__ T Nxis[num_quadrature_pts][dof_per_element];
    __shared__ T B_matrix[num_quadrature_pts][6 * dof_per_element];
    __shared__ T D_matrix[num_quadrature_pts][6 * 6];
    __shared__ T B_T_D_B[num_quadrature_pts][dof_per_element * dof_per_element];
    __shared__ T K_e[dof_per_element * dof_per_element];

    int constexpr spatial_dim_2 = spatial_dim * spatial_dim;
    int pts_offset = k * spatial_dim;
    int J_offset = k * spatial_dim_2;

    if (tid < num_quadrature_pts) {  // tid = quad point index
      wts[tid] =
          Quadrature::template get_quadrature_pt<T>(tid, pts + pts_offset);
    }
    __syncthreads();

    if (tid < num_quadrature_pts * nodes_per_element) {
      for (int j = 0; j < spatial_dim; j++) {
        Nxis[k][i + j * nodes_per_element] = 0.0;
      }
    }

    __syncthreads();

    if (tid < num_quadrature_pts) {
      Basis::template eval_basis_grad_gpu<num_quadrature_pts, dof_per_element>(
          tid, pts, Nxis);
    }

    __syncthreads();

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates
    if (tid < num_quadrature_pts) {
      memset(J + J_offset, 0, spatial_dim_2 * sizeof(T));
    }

    __syncthreads();
    if (tid < num_quadrature_pts) {
      Basis::template eval_grad_gpu<num_quadrature_pts, spatial_dim>(
          tid, pts + pts_offset, element_xloc, J + J_offset);
    }

    __syncthreads();

    if (tid < num_quadrature_pts) {
      int J_offset = tid * spatial_dim_2;
      dets[tid] = det3x3_simple(J + J_offset);
    }
    __syncthreads();

    if (tid < num_quadrature_pts * spatial_dim_2) {
      // Compute the inverse and determinant of the Jacobian matrix
      int k = tid / spatial_dim_2;  // quad index
      int i = tid % spatial_dim_2;  // 0 ~ 8
      inv3x3_gpu(i, J + J_offset, Jinv + J_offset, dets[k]);
    }
    __syncthreads();

    // TODO: parallelize more
    if (tid < num_quadrature_pts) {
      memset(B_matrix[tid], 0, 6 * dof_per_element * sizeof(T));
      Basis::calculate_B_matrix(Jinv + J_offset, pts + pts_offset,
                                B_matrix[tid]);

      memset(D_matrix[tid], 0, 6 * 6 * sizeof(T));
      Basis::template calculate_D_matrix<dof_per_node>(material, D_matrix[tid]);

      memset(B_T_D_B[tid], 0, sizeof(T) * dof_per_element * dof_per_element);
      calculate_B_T_D_B(B_matrix[tid], D_matrix[tid], B_T_D_B[tid]);
    }

    __syncthreads();
    if (tid < num_quadrature_pts) {
      for (int j = 0; j < dof_per_element * dof_per_element; j++) {
        atomicAdd(&K_e[j], wts[tid] * dets[tid] * B_T_D_B[tid][j]);
      }
    }
    __syncthreads();

    // TODO: parallelize
    if (tid == 0) {
      cppimpact_gemv<T, MatOp::NoTrans>(dof_per_element, dof_per_element, 1.0,
                                        K_e, element_dof, 0.0, f_internal);
    }
    __syncthreads();
  }

  static CPPIMPACT_FUNCTION void calculate_B_T_D_B(const T *B_matrix,
                                                   const T *D_matrix,
                                                   T *B_T_D_B) {
    // B_matrix: 6 x N matrix
    // D_matrix: 6 x 6 matrix
    // B_T_D_B: N x N matrix (initialized to zero before calling)
    // N: spatial_dim * nodes_per_element

    const int N = nodes_per_element * spatial_dim;

    for (int k = 0; k < 6; ++k) {
      const T *B_row_k = &B_matrix[k * N];
      for (int l = 0; l < 6; ++l) {
        T Dkl = D_matrix[k * 6 + l];
        const T *B_row_l = &B_matrix[l * N];

        for (int i = 0; i < N; ++i) {
          T Bik_Dkl = B_row_k[i] * Dkl;
          T *B_T_D_B_row = &B_T_D_B[i * N];

          for (int j = 0; j < N; ++j) {
            B_T_D_B_row[j] += Bik_Dkl * B_row_l[j];
          }
        }
      }
    }
  }

  static void calculate_f_internal(const T *element_xloc, const T *element_dof,
                                   T *f_internal,
                                   BaseMaterial<T, dof_per_node> *material) {
    T pt[spatial_dim];
    T K_e[dof_per_element * dof_per_element];
    memset(K_e, 0, sizeof(T) * dof_per_element * dof_per_element);

    for (int i = 0; i < num_quadrature_pts; i++) {
      T weight = Quadrature::template get_quadrature_pt<T>(i, pt);

      // Evaluate the derivative of the spatial dof in the computational
      // coordinates
      T J[spatial_dim * spatial_dim];
      memset(J, 0, spatial_dim * spatial_dim * sizeof(T));
      Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

      // standard basis here
      // Compute the inverse and determinant of the Jacobian matrix
      T Jinv[spatial_dim * spatial_dim];
      memset(Jinv, 0, spatial_dim * spatial_dim * sizeof(T));
      T detJ = inv3x3(J, Jinv);

#ifdef CPPIMPACT_DEBUG_MODE
      if (detJ < 0.0) {
        printf("detJ negative\n");
      }
#endif

      // Compute the B matrix
      // PU used here

      T J_PU[spatial_dim * spatial_dim];
      memset(J_PU, 0, spatial_dim * spatial_dim * sizeof(T));
      Basis::template eval_grad_PU<spatial_dim>(pt, element_xloc, J_PU);

      T Jinv_PU[spatial_dim * spatial_dim];
      memset(Jinv_PU, 0, spatial_dim * spatial_dim * sizeof(T));
      T detJ_PU = inv3x3(J_PU, Jinv_PU);

      T B_matrix[6 * spatial_dim * nodes_per_element];
      memset(B_matrix, 0, 6 * spatial_dim * nodes_per_element * sizeof(T));
      Basis::calculate_B_matrix(Jinv, pt, B_matrix);

      // Compute the material stiffness matrix D
      T D_matrix[6 * 6];
      memset(D_matrix, 0, 6 * 6 * sizeof(T));
      Basis::template calculate_D_matrix<dof_per_node>(material, D_matrix);

      // Compute B^T * D * B
      T B_T_D_B[dof_per_element * dof_per_element];
      memset(B_T_D_B, 0, sizeof(T) * dof_per_element * dof_per_element);
      calculate_B_T_D_B(B_matrix, D_matrix, B_T_D_B);

      // Assemble the element stiffness matrix K_e
      for (int j = 0; j < dof_per_element * dof_per_element; j++) {
        K_e[j] += weight * detJ * B_T_D_B[j];
      }
    }

    cppimpact_gemv<T, MatOp::NoTrans>(dof_per_element, dof_per_element, 1.0,
                                      K_e, element_dof, 0.0, f_internal);
  }

  static T calculate_strain_energy(const T *element_xloc, const T *element_dof,
                                   BaseMaterial<T, dof_per_node> *material) {
    T pt[spatial_dim];
    T K_e[dof_per_element * dof_per_element];
    memset(K_e, 0, sizeof(T) * dof_per_element * dof_per_element);

    T volume = 0.0;

    for (int i = 0; i < num_quadrature_pts; i++) {
      T weight = Quadrature::template get_quadrature_pt<T>(i, pt);

      // Evaluate the derivative of the spatial dof in the computational
      // coordinates
      T J[spatial_dim * spatial_dim];
      memset(J, 0, spatial_dim * spatial_dim * sizeof(T));
      Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

      // Compute the inverse and determinant of the Jacobian matrix
      T Jinv[spatial_dim * spatial_dim];
      memset(Jinv, 0, spatial_dim * spatial_dim * sizeof(T));
      T detJ = inv3x3(J, Jinv);

#ifdef CPPIMPACT_DEBUG_MODE
      if (detJ < 0.0) {
        printf("detJ negative\n");
      }
#endif

      T J_PU[spatial_dim * spatial_dim];
      memset(J_PU, 0, spatial_dim * spatial_dim * sizeof(T));
      Basis::template eval_grad_PU<spatial_dim>(pt, element_xloc, J_PU);

      T Jinv_PU[spatial_dim * spatial_dim];
      memset(Jinv_PU, 0, spatial_dim * spatial_dim * sizeof(T));
      T detJ_PU = inv3x3(J_PU, Jinv);

      // Compute the B matrix
      T B_matrix[6 * spatial_dim * nodes_per_element];
      memset(B_matrix, 0, 6 * spatial_dim * nodes_per_element * sizeof(T));
      Basis::calculate_B_matrix(Jinv, pt, B_matrix);

      // Compute the material stiffness matrix D
      T D_matrix[6 * 6];
      memset(D_matrix, 0, 6 * 6 * sizeof(T));
      Basis::template calculate_D_matrix<dof_per_node>(material, D_matrix);

      // Compute B^T * D * B
      T B_T_D_B[dof_per_element * dof_per_element];
      memset(B_T_D_B, 0, sizeof(T) * dof_per_element * dof_per_element);
      calculate_B_T_D_B(B_matrix, D_matrix, B_T_D_B);

      // Assemble the element stiffness matrix K_e
      for (int j = 0; j < dof_per_element * dof_per_element; j++) {
        K_e[j] += weight * detJ * B_T_D_B[j];
      }
    }

    T Ku[dof_per_element];
    memset(Ku, 0, sizeof(T) * dof_per_element);
    // Multiply K_e * u
    cppimpact_gemv<T, MatOp::NoTrans>(dof_per_element, dof_per_element, 1.0,
                                      K_e, element_dof, 0.0, Ku);

    T W = 0.0;
    for (int j = 0; j < dof_per_element; j++) {
      W += 0.5 * element_dof[j] * Ku[j];
    }

    return W;
  }

  static T calculate_volume(const T *element_xloc, const T *element_dof,
                            BaseMaterial<T, dof_per_node> *material) {
    T pt[spatial_dim];

    T volume = 0.0;

    for (int i = 0; i < num_quadrature_pts; i++) {
      T weight = Quadrature::template get_quadrature_pt<T>(i, pt);

      // Evaluate the derivative of the spatial dof in the computational
      // coordinates
      T J[spatial_dim * spatial_dim];
      Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

      // Compute the inverse and determinant of the Jacobian matrix
      T Jinv[spatial_dim * spatial_dim];
      T detJ = inv3x3(J, Jinv);
      volume += weight * detJ;
    }
    // printf("volume = %f\n", volume);

    return volume;
  }

  static void calculate_strain(const T *element_xloc, const T *element_dof,
                               const T *pt, T *strain,
                               BaseMaterial<T, dof_per_node> *material) {
    T J[spatial_dim * spatial_dim];
    Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

    // Compute the inverse and determinant of the Jacobian matrix
    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3(J, Jinv);

    T J_PU[spatial_dim * spatial_dim];
    memset(J_PU, 0, spatial_dim * spatial_dim * sizeof(T));
    Basis::template eval_grad_PU<spatial_dim>(pt, element_xloc, J_PU);

    T Jinv_PU[spatial_dim * spatial_dim];
    memset(Jinv_PU, 0, spatial_dim * spatial_dim * sizeof(T));
    T detJ_PU = inv3x3(J_PU, Jinv_PU);

    // Compute the B matrix
    T B_matrix[6 * spatial_dim * nodes_per_element];
    memset(B_matrix, 0, 6 * spatial_dim * nodes_per_element * sizeof(T));
    Basis::calculate_B_matrix(Jinv_PU, pt, B_matrix);

    // multiply B*u
    cppimpact_gemv<T, MatOp::NoTrans>(6, dof_per_element, 1.0, B_matrix,
                                      element_dof, 0.0, strain);
  }
};

// explicit instantiation if needed

// using T = double;
// using Basis = TetrahedralBasisQuadratic;
// using Quadrature = TetrahedralQuadrature5pts;
// using Physics = NeohookeanPhysics<T>;
// using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;
// const int nodes_per_element = Basis::nodes_per_element;

// template __global__ void energy_kernel<T, nodes_per_element>(const int
// *element_nodes,
//                                                              const T *xloc,
//                                                              const T *dof,
//                                                              T
//                                                              *total_energy,
//                                                              T *C1, T *D1);

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

// T B_node[6 * 3];  // Temporary storage for a 6x3 block of B for a single
// node T BP_node[3 * 3]; // Temporary storage for the result of B_node^T * P
// for a single node

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