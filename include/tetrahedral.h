#pragma once

#include <iostream>

#include "cppimpact_defs.h"

class TetrahedralQuadrature {
 public:
  static const int num_quadrature_pts = 5;

  template <typename T>
  static CPPIMPACT_FUNCTION T get_quadrature_pt(int k, T pt[]) {
    if (k == 0) {
      pt[0] = 0.25;
      pt[1] = 0.25;
      pt[2] = 0.25;
      return -0.13333333333;
    } else if (k == 1) {
      pt[0] = 0.16666666666;
      pt[1] = 0.16666666666;
      pt[2] = 0.16666666666;
      return 0.075;
    } else if (k == 2) {
      pt[0] = 0.5;
      pt[1] = 0.16666666666;
      pt[2] = 0.16666666666;
      return 0.075;
    } else if (k == 3) {
      pt[0] = 0.16666666666;
      pt[1] = 0.5;
      pt[2] = 0.16666666666;
      return 0.075;
    } else if (k == 4) {
      pt[0] = 0.16666666666;
      pt[1] = 0.16666666666;
      pt[2] = 0.5;
      return 0.075;
    }
    return 0.0;
  }
};

template <typename T>
class TetrahedralBasisQuadratic {
 public:
  static constexpr int spatial_dim = 3;
  static constexpr int nodes_per_element = 10;

  // TODO: this now only uses 30 threads out of 64
  template <int num_quadrature_pts, int dof_per_element>
  static __device__ void eval_basis_grad_gpu_new(
      int tid, int q, const T pts[],
      T Nxis[num_quadrature_pts][dof_per_element]) {
    constexpr T coeffs[30][4] = {
        {4.0, 4.0, 4.0, -3.0},   {4.0, 4.0, 4.0, -3.0},
        {4.0, 4.0, 4.0, -3.0},   {4.0, 0.0, 0.0, -1.0},
        {0.0, 0.0, 0.0, 0.0},    {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},    {0.0, 4.0, 0.0, -1.0},
        {0.0, 0.0, 0.0, 0.0},    {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},    {0.0, 0.0, 4.0, -1.0},
        {-8.0, -4.0, -4.0, 4.0}, {-4.0, 0.0, 0.0, 0.0},
        {-4.0, 0.0, 0.0, 0.0},   {0.0, 4.0, 0.0, 0.0},
        {4.0, 0.0, 0.0, 0.0},    {0.0, 0.0, 0.0, 0.0},
        {0.0, -4.0, 0.0, 0.0},   {-4.0, -8.0, -4.0, 4.0},
        {0.0, -4.0, 0.0, 0.0},   {0.0, 0.0, -4.0, 0.0},
        {0.0, 0.0, -4.0, 0.0},   {-4.0, -4.0, -8.0, 4.0},
        {0.0, 0.0, 4.0, 0.0},    {0.0, 0.0, 0.0, 0.0},
        {4.0, 0.0, 0.0, 0.0},    {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 4.0, 0.0},    {0.0, 4.0, 0.0, 0.0}};

    if (tid < 30) {
      const T* pt = pts + q * spatial_dim;
      Nxis[q][tid] = coeffs[tid][0] * pt[0] + coeffs[tid][1] * pt[1] +
                     coeffs[tid][2] * pt[2] + coeffs[tid][3];
    }
  }

  // Evaluate the gradients of all basis functions on all quadrature points
  template <int num_quadrature_pts, int dof_per_element>
  static __device__ void eval_basis_grad_gpu(
      int tid, const T pts[], T Nxis[num_quadrature_pts][dof_per_element]) {
    // clang-format off
    constexpr T coeffs[30][4] = {
        {4.0, 4.0, 4.0, -3.0},
        {4.0, 4.0, 4.0, -3.0},
        {4.0, 4.0, 4.0, -3.0},

        {4.0, 0.0, 0.0, -1.0},
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},

        {0.0, 0.0, 0.0, 0.0},
        {0.0, 4.0, 0.0, -1.0},
        {0.0, 0.0, 0.0, 0.0},

        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 4.0, -1.0},

        {-8.0, -4.0, -4.0, 4.0},
        {-4.0, 0.0, 0.0, 0.0},
        {-4.0, 0.0, 0.0, 0.0},

        {0.0, 4.0, 0.0, 0.0},
        {4.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},

        {0.0, -4.0, 0.0, 0.0},
        {-4.0, -8.0, -4.0, 4.0},
        {0.0, -4.0, 0.0, 0.0},

        {0.0, 0.0, -4.0, 0.0},
        {0.0, 0.0, -4.0, 0.0},
        {-4.0, -4.0, -8.0, 4.0},

        {0.0, 0.0, 4.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
        {4.0, 0.0, 0.0, 0.0},

        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 4.0, 0.0},
        {0.0, 4.0, 0.0, 0.0}
    };
    // clang-format on

    int Nxi_index = tid / nodes_per_element;
    int Nxi_offset_1 = tid % nodes_per_element;
    int Nxi_offset_2 = Nxi_offset_1 + nodes_per_element;
    int Nxi_offset_3 = Nxi_offset_1 + nodes_per_element * 2;
    if (tid < nodes_per_element * num_quadrature_pts) {
      int pts_offset = (tid % num_quadrature_pts) * spatial_dim;
      // clang-format off
        Nxis[Nxi_index][Nxi_offset_1] =
            coeffs[Nxi_offset_1][0] * pts[pts_offset] + coeffs[Nxi_offset_1][1]
* pts[pts_offset + 1] + coeffs[Nxi_offset_1][2] * pts[pts_offset + 2] +
coeffs[Nxi_offset_1][3]; Nxis[Nxi_index][Nxi_offset_2] = coeffs[Nxi_offset_2][0]
* pts[pts_offset] + coeffs[Nxi_offset_2][1] * pts[pts_offset + 1] +
            coeffs[Nxi_offset_2][2] * pts[pts_offset + 2] +
coeffs[Nxi_offset_2][3]; if (Nxi_offset_3 < 30) { Nxis[Nxi_index][Nxi_offset_3]
= coeffs[Nxi_offset_3][0] * pts[pts_offset] + coeffs[Nxi_offset_3][1] *
pts[pts_offset + 1] + coeffs[Nxi_offset_3][2] * pts[pts_offset + 2] +
coeffs[Nxi_offset_3][3];
      }
      // clang-format on
    }
    __syncthreads();
  }

  static CPPIMPACT_FUNCTION void eval_basis_grad(const T pt[], T Nxi[]) {
    // Corner node derivatives
    Nxi[0] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;
    Nxi[1] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;
    Nxi[2] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;

    Nxi[3] = 4.0 * pt[0] - 1.0;
    Nxi[4] = 0.0;
    Nxi[5] = 0.0;

    Nxi[6] = 0.0;
    Nxi[7] = 4.0 * pt[1] - 1.0;
    Nxi[8] = 0.0;

    Nxi[9] = 0.0;
    Nxi[10] = 0.0;
    Nxi[11] = 4.0 * pt[2] - 1.0;

    // Mid node derivatives
    Nxi[12] = -4.0 * (2.0 * pt[0] + pt[1] + pt[2] - 1.0);
    Nxi[13] = -4.0 * pt[0];
    Nxi[14] = -4.0 * pt[0];

    Nxi[15] = 4.0 * pt[1];
    Nxi[16] = 4.0 * pt[0];
    Nxi[17] = 0.0;

    Nxi[18] = -4.0 * pt[1];
    Nxi[19] = -4.0 * (pt[0] + 2.0 * pt[1] + pt[2] - 1.0);
    Nxi[20] = -4.0 * pt[1];

    Nxi[21] = -4.0 * pt[2];
    Nxi[22] = -4.0 * pt[2];
    Nxi[23] = -4.0 * (pt[0] + pt[1] + 2.0 * pt[2] - 1.0);

    Nxi[24] = 4.0 * pt[2];
    Nxi[25] = 0.0;
    Nxi[26] = 4.0 * pt[0];

    Nxi[27] = 0.0;
    Nxi[28] = 4.0 * pt[2];
    Nxi[29] = 4.0 * pt[1];
  }

  template <int dim>
  static CPPIMPACT_FUNCTION void eval_grad(const T pt[], const T dof[],
                                           T grad[]) {
    T Nxi[spatial_dim * nodes_per_element];
    eval_basis_grad(pt, Nxi);

    for (int k = 0; k < spatial_dim * dim; k++) {
      grad[k] = 0.0;
    }

    for (int k = 0; k < dim; k++) {
      for (int i = 0; i < nodes_per_element; i++) {
        grad[spatial_dim * k] += Nxi[spatial_dim * i] * dof[dim * i + k];
        grad[spatial_dim * k + 1] +=
            Nxi[spatial_dim * i + 1] * dof[dim * i + k];
        grad[spatial_dim * k + 2] +=
            Nxi[spatial_dim * i + 2] * dof[dim * i + k];
      }
    }
  }

#ifdef CPPIMPACT_CUDA_BACKEND
  template <int num_quadrature_pts, int dim>
  static __device__ void eval_grad(int tid, const T pt[], const T dof[],
                                   T grad[]) {
    T Nxi[spatial_dim * nodes_per_element];
    eval_basis_grad(pt, Nxi);
    const int nodes_per_elem_num_quad = nodes_per_element * num_quadrature_pts;

    if (tid < nodes_per_elem_num_quad) {
      for (int k = 0; k < spatial_dim * dim; k++) {
        grad[k] = 0.0;
      }
    }
    __syncthreads();

    if (tid < nodes_per_elem_num_quad) {
      int i = tid / num_quadrature_pts;  // node index
      int k = tid % (dim);               // quadrature index > dim
      if (i < nodes_per_element && k < dim) {
        for (int k = 0; k < dim; k++) {
          // clang-format off
        atomicAdd(&grad[spatial_dim * k],     Nxi[spatial_dim * i]     *
        dof[dim * i + k]); atomicAdd(&grad[spatial_dim * k + 1],
        Nxi[spatial_dim * i + 1] * dof[dim * i + k]);
        atomicAdd(&grad[spatial_dim * k + 2], Nxi[spatial_dim * i + 2] *
        dof[dim * i + k]);
          // clang-format on
        }
      }
    }
  }

  template <int num_quadrature_pts, int dim>
  static __device__ void eval_grad_gpu(int tid, const T pt[], const T dof[],
                                       T grad[], T Nxi[]) {
    const int nodes_per_elem_num_quad = nodes_per_element * num_quadrature_pts;

    if (tid < spatial_dim * dim * num_quadrature_pts) {
      grad[tid] = 0.0;
    }
    // if (tid < num_quadrature_pts) {
    //   for (int k = 0; k < spatial_dim * dim; k++) {
    //     grad[k] = 0.0;
    //   }
    // }
    __syncthreads();

    if (tid < nodes_per_elem_num_quad) {
      int i = tid / num_quadrature_pts;  // node index
      int q = tid % num_quadrature_pts;  // quad index
      int grad_offset = spatial_dim * dim * q;
      if (i < nodes_per_element) {
        for (int k = 0; k < dim; k++) {
          // clang-format off
          atomicAdd(&grad[grad_offset + spatial_dim * k],     Nxi[spatial_dim *
i] *     dof[dim * i + k]); atomicAdd(&grad[grad_offset + spatial_dim * k + 1],
Nxi[spatial_dim * i + 1] * dof[dim * i + k]); atomicAdd(&grad[grad_offset +
spatial_dim * k + 2], Nxi[spatial_dim * i + 2] * dof[dim * i + k]);
          // clang-format on
        }
      }
    }
  }
#endif

  template <int dim>
  static CPPIMPACT_FUNCTION void add_grad(const T pt[], const T coef[],
                                          T res[]) {
    T Nxi[spatial_dim * nodes_per_element];
    eval_basis_grad(pt, Nxi);

    for (int i = 0; i < nodes_per_element; i++) {
      for (int k = 0; k < dim; k++) {
        atomicAdd(&res[dim * i + k],
                  (coef[spatial_dim * k] * Nxi[spatial_dim * i] +
                   coef[spatial_dim * k + 1] * Nxi[spatial_dim * i + 1] +
                   coef[spatial_dim * k + 2] * Nxi[spatial_dim * i + 2]));
      }
    }
  }

  static CPPIMPACT_FUNCTION void eval_basis_PU(const T pt[], T N[]) {
    // PU functions from https://doi.org/10.1016/j.enganabound.2019.04.018
    T L1 = 1.0 - pt[0] - pt[1] - pt[2];
    T L2 = pt[0];
    T L3 = pt[1];
    T L4 = pt[2];
    N[0] = L2 * L2;      // 2 from paper
    N[1] = L3 * L3;      // 3 from paper
    N[2] = L1 * L1;      // 1 from paper
    N[3] = L4 * L4;      // 4 from paper
    N[4] = 2 * L2 * L3;  // 8 from paper
    N[5] = 2 * L1 * L3;  // 6 from paper
    N[6] = 2 * L1 * L2;  // 5 from paper
    N[7] = 2 * L2 * L4;  // 10 from paper
    N[8] = 2 * L3 * L4;  // 9 from paper
    N[9] = 2 * L1 * L4;  // 7 from paper
  }

  static CPPIMPACT_FUNCTION void calculate_B_matrix(const T Jinv[], const T* pt,
                                                    T B[]) {
    // Assuming Nxi is in element coordinates and has dimensions [spatial_dim
    // * nodes_per_element] B matrix should have dimensions [6 *
    // (3*nodes_per_element)], flattened into 1D array
    T Nxi[spatial_dim * nodes_per_element];
    eval_basis_grad(pt, Nxi);

    for (int node = 0; node < nodes_per_element; ++node) {
      T dNdx[spatial_dim];  // Placeholder for gradients in global coordinates
                            // for this node

      // Transform gradients from element to global coordinates using Jinv
      for (int i = 0; i < spatial_dim; ++i) {
        dNdx[i] = 0.0;
        for (int j = 0; j < spatial_dim; ++j) {
          dNdx[i] += Jinv[i * spatial_dim + j] * Nxi[node * spatial_dim + j];
        }
      }

      // Index in the B matrix for the current node
      int idx = 3 * node;

      // Populate the B matrix
      // B[0][idx + 0], B[1][idx + 1], B[2][idx + 2] correspond to normal
      // strains
      B[0 * 3 * nodes_per_element + idx + 0] = dNdx[0];
      B[1 * 3 * nodes_per_element + idx + 1] = dNdx[1];
      B[2 * 3 * nodes_per_element + idx + 2] = dNdx[2];

      // B[3][idx + 0], B[3][idx + 1] correspond to shear strain gamma_xy
      B[3 * 3 * nodes_per_element + idx + 1] = dNdx[2];
      B[3 * 3 * nodes_per_element + idx + 2] = dNdx[1];

      // B[4][idx + 0], B[4][idx + 2] correspond to shear strain gamma_yz
      B[4 * 3 * nodes_per_element + idx + 0] = dNdx[2];
      B[4 * 3 * nodes_per_element + idx + 2] = dNdx[0];

      // B[5][idx + 0], B[5][idx + 2] correspond to shear strain gamma_xz
      B[5 * 3 * nodes_per_element + idx + 0] = dNdx[1];
      B[5 * 3 * nodes_per_element + idx + 1] = dNdx[0];
    }
  }

  template <int dof_per_node>
  static CPPIMPACT_FUNCTION void calculate_D_matrix(
      BaseMaterial<T, dof_per_node>* material, T* D_matrix) {
    // Set diagonal components
    D_matrix[0 * 6 + 0] = D_matrix[1 * 6 + 1] = D_matrix[2 * 6 + 2] =
        1 - material->nu;
    D_matrix[3 * 6 + 3] = D_matrix[4 * 6 + 4] = D_matrix[5 * 6 + 5] =
        1 - 2 * material->nu;
    ;

    // Set off-diagonal components
    D_matrix[0 * 6 + 1] = D_matrix[1 * 6 + 0] = material->nu;
    D_matrix[0 * 6 + 2] = D_matrix[2 * 6 + 0] = material->nu;
    D_matrix[1 * 6 + 2] = D_matrix[2 * 6 + 1] = material->nu;

    // Apply the scalar multiplication
    T scalar = material->E / ((1 + material->nu) * (1 - 2 * material->nu));
    for (int i = 0; i < 36; ++i) {
      D_matrix[i] *= scalar;
    }
  }
};

template <typename T>
class TetrahedralBasisLinear {
 public:
  static constexpr int spatial_dim = 3;
  static constexpr int nodes_per_element = 4;

  static CPPIMPACT_FUNCTION void eval_basis_grad(const T pt[], T Nxi[]) {
    Nxi[0] = -1.0;
    Nxi[1] = -1.0;
    Nxi[2] = -1.0;

    Nxi[3] = 1.0;
    Nxi[4] = 0.0;
    Nxi[5] = 0.0;

    Nxi[6] = 0.0;
    Nxi[7] = 1.0;
    Nxi[8] = 0.0;

    Nxi[9] = 0.0;
    Nxi[10] = 0.0;
    Nxi[11] = 1.0;
  }

  template <int dim>
  static CPPIMPACT_FUNCTION void eval_grad(const T pt[], const T xloc[],
                                           T grad[]) {
    T Nxi[spatial_dim * nodes_per_element];
    eval_basis_grad(pt, Nxi);

    for (int i = 0; i < nodes_per_element; i++) {
      printf("xloc[%d]: %f, %f, %f\n", i, xloc[dim * i], xloc[dim * i + 1],
             xloc[dim * i + 2]);
    }

    // Initialize grad (Jacobian matrix) to zero
    memset(grad, 0, spatial_dim * dim * sizeof(T));

    for (int p = 0; p < dim; p++) {
      for (int q = 0; q < dim; q++) {
        for (int i = 0; i < nodes_per_element; i++) {
          grad[p * dim + q] += Nxi[spatial_dim * i + q] * xloc[dim * i + p];
        }
      }
    }
  }

  template <int dim>
  static CPPIMPACT_FUNCTION void add_grad(const T pt[], const T coef[],
                                          T res[]) {
    T Nxi[spatial_dim * nodes_per_element];
    eval_basis_grad(pt, Nxi);

    for (int i = 0; i < nodes_per_element; i++) {
      for (int k = 0; k < dim; k++) {
        res[dim * i + k] =
            coef[spatial_dim * k] * Nxi[spatial_dim * i] +
            coef[spatial_dim * k + 1] * Nxi[spatial_dim * i + 1] +
            coef[spatial_dim * k + 2] * Nxi[spatial_dim * i + 2];
      }
    }
  }

  static CPPIMPACT_FUNCTION void eval_basis_PU(const T pt[], T N[]) {
    // N[0] = 1.0 - pt[0] - pt[1] - pt[2];  // 2 from paper
    // N[1] = pt[0];                        // 3 from paper
    // N[2] = pt[1];                        // 1 from paper
    // N[3] = pt[2];                        // 4 from paper

    N[0] = 1.0 - pt[0] - pt[1] - pt[2];  // 2 from paper
    N[1] = pt[0];                        // 3 from paper
    N[2] = pt[1];
    N[3] = pt[2];  // 4 from paper
  }

  static CPPIMPACT_FUNCTION void calculate_B_matrix(const T Jinv[], const T* pt,
                                                    T B[]) {
    T Nxi[spatial_dim * nodes_per_element];
    eval_basis_grad(pt, Nxi);

    for (int i = 0; i < spatial_dim * nodes_per_element; ++i) {
      printf("Nxi[%d]: %f\n", i, Nxi[i]);
    }

    for (int node = 0; node < nodes_per_element; ++node) {
      T dNdx[spatial_dim];  // Placeholder for gradients in global coordinates
                            // for this node

      // Transform gradients from element to global coordinates using Jinv
      // TODO: check if Jinv should be transposed
      for (int i = 0; i < spatial_dim; ++i) {
        dNdx[i] = 0.0;
        for (int j = 0; j < spatial_dim; ++j) {
          dNdx[i] += Jinv[j * spatial_dim + i] * Nxi[node * spatial_dim + j];
        }
      }
      printf("dNdx[%d]: %f, %f, %f\n", node, dNdx[0], dNdx[1], dNdx[2]);

      // Index in the B matrix for the current node
      int idx = 3 * node;

      // Populate and print the B matrix
      // B[0][idx + 0], B[1][idx + 1], B[2][idx + 2] correspond to normal
      // strains
      B[0 * 3 * nodes_per_element + idx + 0] = dNdx[0];
      B[1 * 3 * nodes_per_element + idx + 1] = dNdx[1];
      B[2 * 3 * nodes_per_element + idx + 2] = dNdx[2];

      // B[3][idx + 0], B[3][idx + 1] correspond to shear strain gamma_xy
      B[3 * 3 * nodes_per_element + idx + 0] = dNdx[1];
      B[3 * 3 * nodes_per_element + idx + 1] = dNdx[0];

      // B[4][idx + 1], B[4][idx + 2] correspond to shear strain gamma_yz
      B[4 * 3 * nodes_per_element + idx + 1] = dNdx[2];
      B[4 * 3 * nodes_per_element + idx + 2] = dNdx[1];

      // B[5][idx + 0], B[5][idx + 2] correspond to shear strain gamma_xz
      B[5 * 3 * nodes_per_element + idx + 0] = dNdx[2];
      B[5 * 3 * nodes_per_element + idx + 2] = dNdx[0];
    }
  }

  template <int dof_per_node>
  static CPPIMPACT_FUNCTION void calculate_D_matrix(
      BaseMaterial<T, dof_per_node>* material, T* D_matrix) {
    // Fill the matrix
    D_matrix[0 * 6 + 0] = D_matrix[1 * 6 + 1] = D_matrix[2 * 6 + 2] =
        1 - material->nu;
    D_matrix[3 * 6 + 3] = D_matrix[4 * 6 + 4] = D_matrix[5 * 6 + 5] =
        (1 - 2 * material->nu) / 2;

    D_matrix[0 * 6 + 1] = D_matrix[1 * 6 + 0] = material->nu;
    D_matrix[0 * 6 + 2] = D_matrix[2 * 6 + 0] = material->nu;
    D_matrix[1 * 6 + 2] = D_matrix[2 * 6 + 1] = material->nu;

    // Apply the scalar multiplication
    for (int i = 0; i < 36; i++) {
      D_matrix[i] *=
          material->E / ((1 + material->nu) * (1 - 2 * material->nu));
    }
  }
};
