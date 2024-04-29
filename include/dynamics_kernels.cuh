#pragma once

#include "analysis.h"
#include "basematerial.h"
#include "physics.h"
#include "tetrahedral.h"
#include "wall.h"

// Update the system states via time-stepping
template <typename T, int spatial_dim, int nodes_per_element>
__global__ void update(int num_elements, T dt,
                       BaseMaterial<T, spatial_dim>* d_material,
                       Wall<T, 2, TetrahedralBasis<T>>* d_wall,
                       const int* d_element_nodes, const T* d_vel,
                       T* d_global_xloc, T* d_global_acc, T* d_global_dof) {
  using Basis = TetrahedralBasis<T>;
  using Quadrature = TetrahedralQuadrature;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  int constexpr dof_per_element = spatial_dim * nodes_per_element;

  int elem = blockIdx.x;
  if (elem >= num_elements) return;

  __shared__ T element_mass_matrix_diagonals[dof_per_element];
  __shared__ T element_xloc[dof_per_element];
  __shared__ T element_dof[dof_per_element];
  __shared__ T element_acc[dof_per_element];
  __shared__ T element_vel[dof_per_element];
  __shared__ T element_internal_forces[dof_per_element];

  __shared__ int this_element_nodes[nodes_per_element];

  int tid = threadIdx.x;
  if (tid < dof_per_element) {
    element_mass_matrix_diagonals[tid] = 0.0;
    element_xloc[tid] = 0.0;
    element_dof[tid] = 0.0;
    element_acc[tid] = 0.0;
    element_vel[tid] = 0.0;
    element_internal_forces[tid] = 0.0;
  }

  __syncthreads();

  // Get the nodes of this element
  if (tid < nodes_per_element) {
    this_element_nodes[tid] = d_element_nodes[nodes_per_element * elem + tid];
  }
  __syncthreads();

  // Get the element locations
  Analysis::template get_element_dof<
      Analysis::spatial_dim, Analysis::dof_per_element, Analysis::dof_per_node>(
      tid, this_element_nodes, d_global_xloc, element_xloc);

  // Get the element degrees of freedom
  Analysis::template get_element_dof<
      Analysis::spatial_dim, Analysis::dof_per_element, Analysis::dof_per_node>(
      tid, this_element_nodes, d_global_dof, element_dof);

  __syncthreads();
  // Calculate element mass matrix
  Analysis::element_mass_matrix(tid, d_material->rho, element_xloc, element_dof,
                                element_mass_matrix_diagonals);

  T Mr_inv = 0.0;
  if (tid < dof_per_element) {
    Mr_inv = 1.0 / element_mass_matrix_diagonals[tid];
  }

  Analysis::calculate_f_internal(tid, element_xloc, element_dof,
                                 element_internal_forces, d_material);
  __syncthreads();

  // Calculate element acceleration
  if (tid < dof_per_element) {
    element_acc[tid] = Mr_inv * (-element_internal_forces[tid]);
  }

#if 0
  // assemble global mass matrix
  for (int j = 0; j < nodes_per_element; j++) {
    int node = this_element_nodes[j];

    global_mass[3 * node] +=  element_mass_matrix_diagonals[3 * j];
    global_mass[3 * node + 1] += element_mass_matrix_diagonals[3 * j + 1];
    global_mass[3 * node + 2] += element_mass_matrix_diagonals[3 * j + 2];
  }
#endif
}