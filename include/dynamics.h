#pragma once
#include <iostream>
#include <numeric>

#include "mesh.h"
#include "basematerial.h"

template <typename T, class Basis, class Analysis>
class Dynamics
{
public:
  Mesh<T> *mesh;

  int *reduced_nodes;
  int reduced_dofs_size;
  int ndof;
  static constexpr int nodes_per_element = Basis::nodes_per_element;
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int dof_per_node = spatial_dim;
  BaseMaterial<T, dof_per_node> *material;

  T *vel;

  Dynamics(Mesh<T> *input_mesh, BaseMaterial<T, dof_per_node> *input_material)
      : mesh(input_mesh), material(input_material), reduced_nodes(nullptr), reduced_dofs_size(0), vel(nullptr) { ndof = mesh->num_nodes * dof_per_node; }

  ~Dynamics()
  {
    delete[] reduced_nodes;
    delete[] vel;
  }

  // Initialize the body. Move the mesh origin to init_position and give all nodes init_velocity.
  void initialize(T init_position[dof_per_node], T init_velocity[dof_per_node])
  {

    std::cout << "ndof: " << ndof << std::endl;
    vel = new T[ndof];
    for (int i = 0; i < mesh->num_nodes; i++)
    {
      vel[3 * i] = init_velocity[0];
      vel[3 * i + 1] = init_velocity[1];
      vel[3 * i + 2] = init_velocity[2];

      mesh->xloc[3 * i] = mesh->xloc[3 * i] + init_position[0];
      mesh->xloc[3 * i + 1] = mesh->xloc[3 * i + 1] + init_position[1];
      mesh->xloc[3 * i + 2] = mesh->xloc[3 * i + 2] + init_position[2];
    }
  }

  void get_node_global_dofs(const int node_idx, int *global_dof_idx)
  {
    for (int i = 0; i < dof_per_node; i++)
    {
      global_dof_idx[i] = node_idx * dof_per_node + i;
    }
  }

  // This function is used to get the reduced degrees of freedom (DOFs) of the system.
  // It first initializes the reduced DOFs to be all DOFs, then removes the fixed DOFs.
  void get_reduced_nodes()
  {
    // Safe deletion in case it was already allocated
    delete[] reduced_nodes;

    // Allocate memory for reduced DOFs
    reduced_nodes = new int[mesh->num_nodes];

    for (int i = 0; i < mesh->num_nodes; i++)
    {
      reduced_nodes[i] = mesh->element_nodes[i];
    }

    // Loop over all fixed nodes
    for (int i = 0; i < mesh->num_fixed_nodes; i++)
    {
      // Get the value of the fixed node
      int fixed_node_value = mesh->fixed_nodes[i];

      // Loop over reduced_nodes and mark matching nodes as -1
      for (int j = 0; j < mesh->num_nodes; j++)
      {
        if (reduced_nodes[j] == fixed_node_value)
        {
          reduced_nodes[j] = -1; // Mark this node as fixed
        }
      }
    }
  }

  void assemble_diagonal_mass_vector(T *mass_vector)
  {
    /*
    Assemble the global mass matrix in diagonal form.
    Steps
    1. Calculate element mass matrix, diagonalize and lump to nodes.
    2. Collect nodal mass mass matrix into a vector.
    3. Return reduced mass matrix based on BC.
    */

    printf("Getting lumped mass \n");

    T mass = 0.0;
  }

  // void update_mesh(int new_num_elements, int new_num_nodes,
  //                  int* new_element_nodes, T* new_xloc) {
  //   num_elements = new_num_elements;
  //   num_nodes = new_num_nodes;
  //   element_nodes = new_element_nodes;
  //   xloc = new_xloc;
  // }

  void solve(double dt)
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

    // ------------------- Initialization -------------------
    constexpr int dof_per_element = nodes_per_element * spatial_dim;
    printf("Solving dynamics\n");

    // Test matrix
    const T element_density = material->rho_density;
    T dof[ndof];
    for (int i = 0; i < ndof; i++)
    {
      dof[i] = 0.01 * rand() / RAND_MAX;
    }

    int *element_nodes = mesh->element_nodes;
    T *xloc = mesh->xloc;

    T element_mass_matrix_diagonals[dof_per_element];
    T element_xloc[dof_per_element];
    T element_dof[dof_per_element];
    T element_forces[dof_per_element];
    T element_vel[dof_per_element];
    T element_acc[dof_per_element];
    int *this_element_nodes;
    double time = 0.0;
    // T element_density;

    for (int k = 0; k < dof_per_element; k++)
    {
      element_mass_matrix_diagonals[k] = 0.0;
      element_xloc[k] = 0.0;
      element_dof[k] = 0.0;
      element_forces[k] = 0.0;
      element_acc[k] = 0.0;
    }

    // Loop over all elements
    for (int i = 0; i < mesh->num_elements; i++)
    {
      // element_density = element_densities[i];

      this_element_nodes = &element_nodes[nodes_per_element * i];

      // Get the element locations
      Analysis::template get_element_dof<spatial_dim>(this_element_nodes, xloc, element_xloc);
      // Get the element degrees of freedom
      Analysis::template get_element_dof<spatial_dim>(this_element_nodes, dof, element_dof);
      // Get the element velocities
      Analysis::template get_element_dof<spatial_dim>(this_element_nodes, vel, element_vel);

      Analysis::element_mass_matrix(element_density, element_xloc, element_dof, element_mass_matrix_diagonals, i);

      T Mr_inv[dof_per_element];
      for (int k = 0; k < dof_per_element; k++)
      {
        Mr_inv[k] = 1.0 / element_mass_matrix_diagonals[k];
      }

      // get_reduced_dofs(); // Reduced Dofs currently not implemented

      // update_node_positions();
      // assemble_diagonal_mass_vector();
    }

    //------------------- End of Initialization -------------------
    // ------------------- Start of Time Loop -------------------
  }
};