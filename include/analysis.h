#pragma once

#include "physics.h"
#include "tetrahedral.h"
#include <cuda_runtime.h>
#include "basematerial.h"
#include <cblas.h>

template <typename T, class Basis, class Quadrature, class Physics>
class FEAnalysis;
using T = double;
using Basis = TetrahedralBasis<T>;
using Quadrature = TetrahedralQuadrature;
using Physics = NeohookeanPhysics<T>;
// using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

void extract_B_node_block(const T *B, T *B_node, int node, int num_nodes)
{
  for (int i = 0; i < 6; ++i)
  { // 6 rows for strain components
    for (int j = 0; j < 3; ++j)
    { // 3 columns for each node's x, y, z displacement derivatives
      B_node[i * 3 + j] = B[i * (3 * num_nodes) + (3 * node + j)];
    }
  }
}

void multiply_BT_node_P(const T *B_node, const T *P, T *BP_node)
{
  for (int i = 0; i < 3; ++i)
  { // Iterate over rows of B^T (and resulting matrix)
    for (int j = 0; j < 3; ++j)
    {                         // Iterate over columns of P (and resulting matrix)
      BP_node[i * 3 + j] = 0; // Initialize the element to 0
      for (int k = 0; k < 6; ++k)
      { // Iterate over columns of B^T (rows of B_node)
        // Accumulate the product
        BP_node[i * 3 + j] += B_node[k * 3 + i] * P[k * 3 + j];
      }
    }
  }
}

template <typename T, const int nodes_per_element, const int dof_per_node, const int spatial_dim>
__global__ void energy_kernel(const int *element_nodes,
                              const T *xloc, const T *dof, T *total_energy, T *C1, T *D1)
{
  using Analysis = FEAnalysis<T, Basis, TetrahedralQuadrature, NeohookeanPhysics<T>>;
  int element_index = blockIdx.x;
  int thread_index = threadIdx.x;
  const int dof_per_element = dof_per_node * nodes_per_element;

  __shared__ T elem_energy;
  __shared__ T element_xloc[dof_per_element];
  __shared__ T element_dof[dof_per_element];
  elem_energy = 0.0;

  // Get the element node locations
  if (thread_index == 0)
  {

    Analysis::get_element_dof<spatial_dim>(
        &element_nodes[nodes_per_element * element_index], xloc, element_xloc);
    // printf("test 3 \n");
    // Get the element degrees of freedom

    Analysis::get_element_dof<spatial_dim>(
        &element_nodes[nodes_per_element * element_index], dof, element_dof);
  }

  T pt[spatial_dim];
  T weight = TetrahedralQuadrature::get_quadrature_pt<T>(thread_index, pt);

  // Evaluate the derivative of the spatial dof in the computational
  // coordinates
  T J[spatial_dim * spatial_dim];
  Basis::eval_grad<T, spatial_dim>(pt, element_xloc, J);

  // Evaluate the derivative of the dof in the computational coordinates
  T grad[spatial_dim * spatial_dim];
  Basis::eval_grad<T, dof_per_node>(pt, element_dof, grad);
  // Add the energy contributions
  __syncthreads();
  atomicAdd(&elem_energy, Physics::energy(weight, J, grad, *C1, *D1));
  __syncthreads();
  if (thread_index == 0)
  {
    // printf("block %i, quad %i, energy %f, grad %f, element_dof %f  \n",
    //        element_index, j, elem_energy, grad[0], element_dof[0]);
    atomicAdd(total_energy, elem_energy);
  }
}

template <typename T, const int nodes_per_element, const int dof_per_node, const int spatial_dim>
__global__ void residual_kernel(int element_nodes[], const T xloc[], const T dof[],
                                T res[], T *C1, T *D1)
{
  using Analysis = FEAnalysis<T, Basis, TetrahedralQuadrature, NeohookeanPhysics<T>>;
  const int dof_per_element = dof_per_node * nodes_per_element;

  __shared__ T element_res[dof_per_element];
  __shared__ T element_xloc[dof_per_element];
  __shared__ T element_dof[dof_per_element];
  int element_index = blockIdx.x;

  // Parallel initialization of element_res
  for (int i = threadIdx.x; i < dof_per_element; i += blockDim.x)
  {
    element_res[i] = 0.0;
  }

  __syncthreads();

  // Get the element node locations
  if (threadIdx.x == 0)
  {

    Analysis::get_element_dof<spatial_dim>(
        &element_nodes[nodes_per_element * element_index], xloc, element_xloc);
    // printf("test 3 \n");
    // Get the element degrees of freedom

    Analysis::get_element_dof<spatial_dim>(
        &element_nodes[nodes_per_element * element_index], dof, element_dof);
  }

  __syncthreads();

  T pt[spatial_dim];
  T weight = TetrahedralQuadrature::get_quadrature_pt<T>(threadIdx.x, pt);

  // Evaluate the derivative of the spatial dof in the computational
  // coordinates
  T J[spatial_dim * spatial_dim];
  Basis::eval_grad<T, spatial_dim>(pt, element_xloc, J);

  // Evaluate the derivative of the dof in the computational coordinates
  T grad[dof_per_node * spatial_dim];
  Basis::eval_grad<T, dof_per_node>(pt, element_dof, grad);

  // Evaluate the residuals at the quadrature points
  T coef[dof_per_node * spatial_dim];
  Physics::residual(weight, J, grad, coef, *C1, *D1);
  __syncthreads();

  // Add the contributions to the element residual
  Basis::add_grad<T, dof_per_node>(pt, coef, element_res);

  __syncthreads();
  if (threadIdx.x == 0)
  {
    Analysis::add_element_res<dof_per_node>(&element_nodes[nodes_per_element * element_index],
                                            element_res, res);
  }
}

template <typename T, class Basis, class Quadrature, class Physics>
class FEAnalysis
{
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
                              T element_dof[])
  {
    for (int j = 0; j < nodes_per_element; j++)
    {
      int node = nodes[j];
      for (int k = 0; k < dof_per_node; k++, element_dof++)
      {
        element_dof[0] = dof[ndof * node + k];
      }
    }
  }

  template <int ndof>
  static void add_element_res(const int nodes[], const T element_res[],
                              T *res)
  {
    for (int j = 0; j < nodes_per_element; j++)
    {
      int node = nodes[j];
      for (int k = 0; k < spatial_dim; k++, element_res++)
      {
        atomicAdd(&res[ndof * node + k], element_res[0]);
      }
    }
  }

  static T energy(int num_elements, int element_nodes[], T xloc[], T dof[], const int num_nodes, T C1, T D1)
  {
    cudaError_t err;
    T total_energy = 0.0;

    const int threads_per_block = num_quadrature_pts;
    const int num_blocks = num_elements;

    T *d_total_energy;
    cudaMalloc(&d_total_energy, sizeof(T));
    cudaMemset(d_total_energy, 0.0, sizeof(T));

    int *d_element_nodes;
    cudaMalloc(&d_element_nodes, sizeof(int) * num_elements * nodes_per_element);
    cudaMemcpy(d_element_nodes, element_nodes, sizeof(int) * num_elements * nodes_per_element, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("CUDA error memory allocation: %s\n", cudaGetErrorString(err));
    }

    T *d_xloc;
    cudaMalloc(&d_xloc, sizeof(T) * num_nodes * spatial_dim);
    cudaMemcpy(d_xloc, xloc, sizeof(T) * num_nodes * spatial_dim, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("CUDA error memory allocation: %s\n", cudaGetErrorString(err));
    }

    T *d_dof;
    cudaMalloc(&d_dof, sizeof(T) * num_nodes * dof_per_node);
    cudaMemcpy(d_dof, dof, sizeof(T) * num_nodes * dof_per_node, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("CUDA error memory allocation: %s\n", cudaGetErrorString(err));
    }

    T *d_C1;
    T *d_D1;
    cudaMalloc(&d_C1, sizeof(T));
    cudaMalloc(&d_D1, sizeof(T));

    cudaMemcpy(d_C1, &C1, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D1, &D1, sizeof(T), cudaMemcpyHostToDevice);

    printf("Total Elements: %i \n", num_elements);
    printf("Num Blocks: %i \n", num_blocks);
    printf("Total Threads: %i \n", num_blocks * threads_per_block);

    energy_kernel<T, nodes_per_element, dof_per_node, spatial_dim><<<num_blocks, threads_per_block>>>(d_element_nodes,
                                                                                                      d_xloc, d_dof, d_total_energy, d_C1, d_D1);
    cudaDeviceSynchronize();
    cudaMemcpy(&total_energy, d_total_energy, sizeof(T), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_total_energy);
    cudaFree(d_element_nodes);
    cudaFree(d_xloc);
    cudaFree(d_dof);
    cudaFree(d_C1);
    cudaFree(d_D1);
    return total_energy;
  }

  static void residual(int num_elements, int num_nodes,
                       const int element_nodes[], const T xloc[], const T dof[],
                       T res[], T C1, T D1)
  {
    cudaError_t err;
    const int threads_per_block = num_quadrature_pts;
    const int num_blocks = num_elements;

    T *d_res;
    cudaMalloc(&d_res, num_nodes * spatial_dim * sizeof(T));
    cudaMemset(d_res, 0, num_nodes * spatial_dim * sizeof(T));

    int *d_element_nodes;
    cudaMalloc(&d_element_nodes, sizeof(int) * num_elements * nodes_per_element);
    cudaMemcpy(d_element_nodes, element_nodes, sizeof(int) * num_elements * nodes_per_element, cudaMemcpyHostToDevice);

    T *d_xloc;
    cudaMalloc(&d_xloc, sizeof(T) * num_nodes * spatial_dim);
    cudaMemcpy(d_xloc, xloc, sizeof(T) * num_nodes * spatial_dim, cudaMemcpyHostToDevice);

    T *d_dof;
    cudaMalloc(&d_dof, sizeof(T) * num_nodes * dof_per_node);
    cudaMemcpy(d_dof, dof, sizeof(T) * num_nodes * dof_per_node, cudaMemcpyHostToDevice);

    T *d_C1;
    T *d_D1;
    cudaMalloc(&d_C1, sizeof(T));
    cudaMalloc(&d_D1, sizeof(T));

    cudaMemcpy(d_C1, &C1, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D1, &D1, sizeof(T), cudaMemcpyHostToDevice);

    printf("Total Elements: %i \n", num_elements);
    printf("Num Blocks: %i \n", num_blocks);
    printf("Total Threads: %i \n", num_blocks * threads_per_block);

    residual_kernel<T, nodes_per_element, dof_per_node, spatial_dim><<<num_blocks, threads_per_block>>>(d_element_nodes, d_xloc, d_dof,
                                                                                                        d_res, d_C1, d_D1);
    cudaDeviceSynchronize();
    cudaMemcpy(res, d_res, num_nodes * spatial_dim * sizeof(T), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_element_nodes);
    cudaFree(d_xloc);
    cudaFree(d_dof);
    cudaFree(d_C1);
    cudaFree(d_D1);
    cudaFree(d_res);
  }

  static void jacobian_product(int num_elements,
                               const int element_nodes[], const T xloc[],
                               const T dof[], const T direct[], T res[], const T C1, const T D1)
  {
    for (int i = 0; i < num_elements; i++)
    {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_dof<spatial_dim>(&element_nodes[nodes_per_element * i], xloc,
                                   element_xloc);

      // Get the element degrees of freedom
      T element_dof[dof_per_element];
      get_element_dof<dof_per_node>(&element_nodes[nodes_per_element * i], dof,
                                    element_dof);

      // Get the element directions for the Jacobian-vector product
      T element_direct[dof_per_element];
      get_element_dof<dof_per_node>(&element_nodes[nodes_per_element * i],
                                    direct, element_direct);

      // Create the element residual
      T element_res[dof_per_element];
      for (int j = 0; j < dof_per_element; j++)
      {
        element_res[j] = 0.0;
      }

      for (int j = 0; j < num_quadrature_pts; j++)
      {
        T pt[spatial_dim];
        T weight = Quadrature::template get_quadrature_pt<T>(j, pt);

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        T J[spatial_dim * spatial_dim];
        Basis::template eval_grad<T, spatial_dim>(pt, element_xloc, J);

        // Evaluate the derivative of the dof in the computational coordinates
        T grad[dof_per_node * spatial_dim];
        Basis::template eval_grad<T, dof_per_node>(pt, element_dof, grad);

        // Evaluate the derivative of the direction in the computational
        // coordinates
        T grad_direct[dof_per_node * spatial_dim];
        Basis::template eval_grad<T, dof_per_node>(pt, element_direct,
                                                   grad_direct);

        // Evaluate the residuals at the quadrature points
        T coef[dof_per_node * spatial_dim];
        Physics::jacobian(weight, J, grad, grad_direct, coef, C1, D1);

        // Add the contributions to the element residual
        Basis::template add_grad<T, dof_per_node>(pt, coef, element_res);
      }

      add_element_res<dof_per_node>(&element_nodes[nodes_per_element * i],
                                    element_res, res);
    }
  }

  static void element_mass_matrix(const T element_density, const T *element_xloc, const T *element_dof, T *element_mass_matrix_diagonals, int element_index)
  {
    for (int i = 0; i < nodes_per_element; i++)
    {
      T m_i = 0.0;
      for (int k = 0; k < num_quadrature_pts; k++)
      {
        T pt[spatial_dim];
        T weight = Quadrature::get_quadrature_pt(k, pt);
        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        T J[spatial_dim * spatial_dim];
        Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

        // Evaluate the derivative of the dof in the computational coordinates
        T grad[spatial_dim * spatial_dim];
        Basis::template eval_grad<dof_per_node>(pt, element_dof, grad);

        // Compute the inverse and determinant of the Jacobian matrix
        T Jinv[spatial_dim * spatial_dim];
        T detJ = inv3x3(J, Jinv);

        // Compute the derformation gradient
        T F[spatial_dim * spatial_dim];
        mat3x3MatMult(grad, Jinv, F);
        F[0] += 1.0;
        F[4] += 1.0;
        F[8] += 1.0;

        // Compute the invariants
        T detF = det3x3(F);
        T N[nodes_per_element];
        Basis::eval_basis_PU(pt, N);
        m_i += N[i] * weight * detJ * element_density;
      }
      element_mass_matrix_diagonals[3 * i] = m_i;
      element_mass_matrix_diagonals[3 * i + 1] = m_i;
      element_mass_matrix_diagonals[3 * i + 2] = m_i;
    }
  }

  static void calculate_f_internal(const T *element_xloc, const T *element_dof, T *f_internal, BaseMaterial<T, dof_per_node> *material)
  {
    T pt[spatial_dim];
    for (int i = 0; i < num_quadrature_pts; i++)
    {
      T weight = Quadrature::template get_quadrature_pt<T>(i, pt);
      // Evaluate the derivative of the spatial dof in the computational
      // coordinates
      T J[spatial_dim * spatial_dim];
      Basis::template eval_grad<spatial_dim>(pt, element_xloc, J);

      // Evaluate the derivative of the dof in the computational coordinates
      T grad[dof_per_node * spatial_dim];
      Basis::template eval_grad<dof_per_node>(pt, element_dof, grad);

      // Compute the inverse and determinant of the Jacobian matrix
      T Jinv[spatial_dim * spatial_dim];
      T detJ = inv3x3(J, Jinv);

      // Compute the deformation gradient F
      T F[spatial_dim * spatial_dim];
      mat3x3MatMult(grad, Jinv, F);
      F[0] += 1.0;
      F[4] += 1.0;
      F[8] += 1.0;

      T Finv[spatial_dim * spatial_dim];
      T FinvT[spatial_dim * spatial_dim];
      T Fdet = inv3x3(F, Finv);
      transpose3x3(Finv, FinvT);

      T J_neo = det3x3(F); // Compute determinant of F to get J
      T lnJ = std::log(detJ);

      T P[spatial_dim * spatial_dim];

      // Initialize P values to 0.0
      for (int j = 0; j < spatial_dim * spatial_dim; ++j)
      {
        P[j] = 0.0;
      }

      T mu = material->E / (2 * (1 + material->nu));
      T lambda = material->E * material->nu / ((1 + material->nu) * (1 - 2 * material->nu));

      // Compute P: First Piola-Kirchhoff stress tensor
      for (int j = 0; j < 9; ++j)
      {                                // For each component in the 3x3 matrix
        P[j] = mu * (F[j] - FinvT[j]); // mu * (F - F^{-T})
        if (j % 4 == 0)
        {                                  // Diagonal components
          P[j] += lambda * lnJ * FinvT[j]; // Add lambda * ln(J) * F^{-T} term
        }
      }

      T B_matrix[6 * 3 * nodes_per_element];
      for (int j = 0; j < 6 * 3 * nodes_per_element; ++j)
      {
        B_matrix[j] = 0.0;
      }

      Basis::calculate_B_matrix(Jinv, grad, B_matrix);

      T B_node[6 * 3];  // Temporary storage for a 6x3 block of B for a single node
      T BP_node[3 * 3]; // Temporary storage for the result of B_node^T * P for a single node

      // Initialize B_node and BP_node to -5.0
      for (int j = 0; j < 18; ++j)
      {
        B_node[j] = 0.0;
      }
      for (int j = 0; j < 9; ++j)
      {
        BP_node[j] = 0.0;
      }

      for (int node = 0; node < nodes_per_element; ++node)
      {
        // Extract the 6x3 block for this node from B
        extract_B_node_block(B_matrix, B_node, node, nodes_per_element);

        // Perform the multiplication B_node^T * P
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    3, 3, 6,
                    1.0, B_node, 3,
                    P, 3,
                    0.0, BP_node, 3);

        // Accumulate the result into f_internal for this node
        for (int j = 0; j < 3; ++j)
        {
          for (int k = 0; k < 3; ++k)
          {
            f_internal[node * 3 + j] += BP_node[j * 3 + k] * weight * detJ; // Consider the quadrature weight and detJ
          }
        }
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

// template __global__ void energy_kernel<T, nodes_per_element>(const int *element_nodes,
//                                                              const T *xloc, const T *dof, T *total_energy, T *C1, T *D1);