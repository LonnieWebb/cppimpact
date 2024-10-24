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

    if (tid == 0) {
      memset(pts, 0, num_quadrature_pts * spatial_dim * sizeof(T));
      memset(dets, 0, num_quadrature_pts * sizeof(T));
      memset(wts, 0, num_quadrature_pts * sizeof(T));
      memset(J, 0, num_quadrature_pts * spatial_dim * spatial_dim * sizeof(T));
      memset(Jinv, 0,
             num_quadrature_pts * spatial_dim * spatial_dim * sizeof(T));
      memset(K_e, 0, sizeof(T) * dof_per_element * dof_per_element);
    }

    if (tid < num_quadrature_pts) {
      memset(Nxis[tid], 0, dof_per_element * sizeof(T));
      memset(B_matrix[tid], 0, 6 * dof_per_element * sizeof(T));
      memset(D_matrix[tid], 0, 6 * 6 * sizeof(T));
      memset(B_T_D_B[tid], 0, sizeof(T) * dof_per_element * dof_per_element);
    }
    __syncthreads();

    int constexpr spatial_dim_2 = spatial_dim * spatial_dim;
    int pts_offset = k * spatial_dim;
    int J_offset = k * spatial_dim_2;

    if (tid < num_quadrature_pts) {  // tid = quad point index
      wts[tid] =
          Quadrature::template get_quadrature_pt<T>(tid, pts + pts_offset);
    }
    __syncthreads();

    if (tid < num_quadrature_pts) {
      Basis::template eval_grad_gpu<num_quadrature_pts, spatial_dim>(
          tid, pts + pts_offset, element_xloc, J + J_offset);
    }

    __syncthreads();

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates

    if (tid < num_quadrature_pts) {
      int J_offset = tid * spatial_dim_2;
      dets[tid] = det3x3_simple(J + J_offset);
    }
    __syncthreads();

    if (tid == 0) {
      // printf("J = [%f, %f, %f, %f, %f, %f, %f, %f, %f], det = %f\n",
      //        J[0 + 1 * spatial_dim_2], J[1 + 1 * spatial_dim_2],
      //        J[2 + 1 * spatial_dim_2], J[3 + 1 * spatial_dim_2],
      //        J[4 + 1 * spatial_dim_2], J[5 + 1 * spatial_dim_2],
      //        J[6 + 1 * spatial_dim_2], J[7 + J_offset], J[8 + J_offset],
      //        dets[1]);

      for (int detnum = 0; detnum < 5; detnum++) {
        if (dets[detnum] <= 0) {
          printf("det[%d] = %f\n", detnum, dets[detnum]);
        }
      }
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
      Basis::calculate_B_matrix(Jinv + J_offset, pts + pts_offset,
                                B_matrix[tid]);
    }
    __syncthreads();

    if (tid < num_quadrature_pts) {
      Basis::template calculate_D_matrix<dof_per_node>(material, D_matrix[tid]);
    }

    __syncthreads();
    if (tid < num_quadrature_pts) {
      calculate_B_T_D_B(B_matrix[tid], D_matrix[tid], B_T_D_B[tid]);
    }

    __syncthreads();

    if (tid < num_quadrature_pts) {
      for (int j = 0; j < dof_per_element * dof_per_element; j++) {
        atomicAdd(&K_e[j], wts[tid] * dets[tid] * B_T_D_B[tid][j]);
      }
    }
    __syncthreads();

    if (tid == 0) {
      for (int mkmk = 0; mkmk < 12; mkmk++) {
        // printf("B_T_D_B = %f\n", element_xloc[mkmk]);
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

  static CPPIMPACT_FUNCTION void calculate_stress_strain(
      const T *element_xloc, const T *element_dof, const T *pt, T *strain,
      T *stress, BaseMaterial<T, dof_per_node> *material) {
    T J[spatial_dim * spatial_dim];
    Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

    // Compute the inverse and determinant of the Jacobian matrix
    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3(J, Jinv);

    // Compute the B matrix
    T B_matrix[6 * spatial_dim * nodes_per_element];
    memset(B_matrix, 0, 6 * spatial_dim * nodes_per_element * sizeof(T));
    Basis::calculate_B_matrix(Jinv, pt, B_matrix);

    // multiply B*u
    cppimpact_gemv<T, MatOp::NoTrans>(6, dof_per_element, 1.0, B_matrix,
                                      element_dof, 0.0, strain);

    T D_matrix[6 * 6];
    memset(D_matrix, 0, 6 * 6 * sizeof(T));
    Basis::calculate_D_matrix(material, D_matrix);

    cppimpact_gemv<T, MatOp::NoTrans>(6, 6, 1.0, D_matrix, strain, 0.0, stress);
  }
};
