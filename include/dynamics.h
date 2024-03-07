#pragma once
#include <iostream>
#include <numeric>

#include "mesh.h"

template <typename T, int dof_per_node>
class Dynamics
{
public:
  Mesh<T> *mesh;
  int *reduced_dofs;
  int reduced_dofs_size;

  Dynamics(Mesh<T> *input_mesh)
      : mesh(input_mesh), reduced_dofs(nullptr), reduced_dofs_size(0) {}

  ~Dynamics() { delete[] reduced_dofs; }

  void get_node_global_dofs(const int node_idx, int *global_dof_idx)
  {
    for (int i = 0; i < dof_per_node; i++)
    {
      global_dof_idx[i] = node_idx * dof_per_node + i;
    }
  }

  void get_reduced_dofs()
  {
    int size = mesh->num_nodes * dof_per_node;
    delete[] reduced_dofs; // Safe deletion in case it was already allocated
    reduced_dofs = new int[size];
    reduced_dofs_size = size;

    for (int i = 0; i < mesh->num_nodes * dof_per_node; i++)
    {
      reduced_dofs[i] = i;
    }

    int global_dof_idx[dof_per_node];
    for (int i = 0; i < mesh->num_fixed_nodes; i++)
    {
      int fixed_node_idx = mesh->fixed_nodes[i];

      get_node_global_dofs(fixed_node_idx, global_dof_idx);
      for (int j = 0; j < dof_per_node; j++)
      {
        int dof_idx = global_dof_idx[j];
        reduced_dofs[dof_idx] = -1;
      }
    }

    int new_size = dof_per_node * (mesh->num_nodes - mesh->num_fixed_nodes);
    int *new_reduced_dofs = new int[new_size];

    int index = 0;
    for (int i = 0; i < size; i++)
    {
      if (reduced_dofs[i] != -1)
      {
        new_reduced_dofs[index++] = reduced_dofs[i];
      }
    }
    delete[] reduced_dofs;
    reduced_dofs = new_reduced_dofs;
    reduced_dofs_size = new_size;
  }

  // void update_mesh(int new_num_elements, int new_num_nodes,
  //                  int* new_element_nodes, T* new_xloc) {
  //   num_elements = new_num_elements;
  //   num_nodes = new_num_nodes;
  //   element_nodes = new_element_nodes;
  //   xloc = new_xloc;
  // }

  void solve()
  {
    // Perform a dynamic analysis. The algorithm is staggered as follows:
    // This assumes that the initial u, v, a and fext are already initialized
    // at nodes.

    // Given U0 and V0,
    // a. A0 = (Fext - Fint(U0))/M
    // b. Stagger V0.5 = V0 +dt/2*a0

    // Now start the loop
    // 1. Compute U1 = U +dt*V0.5
    // 2. Compute A1 = (Fext - Fint(U1)/M
    // 3. Compute V1.5 = V0.5 + A1*dt
    // 3. Compute V1 = V1.5 - dt/2 * a1
    // 4. Loop back to 1.

    // This scheme is common among various commercial solvers,
    // and hence, preferrable.
    get_reduced_dofs();
  }
};