#pragma once
#include <vector>

#include "basematerial.h"
#include "wall.h"

// Update the system states via time-stepping
template <typename T, int spatial_dim, int nodes_per_element, class Basis,
          class Analysis>
void update(int num_nodes, int num_elements, int ndof, T dt, T element_density,
            BaseMaterial<T, spatial_dim> *material, Wall<T, 2, Basis> *wall,
            const int *element_nodes, const T *vel, const T *global_xloc,
            T *global_acc, T *global_dof, T *global_mass) {
  int constexpr dof_per_element = spatial_dim * nodes_per_element;

  // Zero-out states
  memset(global_acc, 0, sizeof(T) * ndof);
  memset(global_dof, 0, sizeof(T) * ndof);
  memset(global_mass, 0, sizeof(T) * ndof);

  // Allocate element quantities
  std::vector<T> element_mass_matrix_diagonals(dof_per_element);
  std::vector<T> element_xloc(dof_per_element);
  std::vector<T> element_dof(dof_per_element);
  std::vector<T> element_acc(dof_per_element);
  std::vector<T> element_internal_forces(dof_per_element);
  std::vector<int> this_element_nodes(nodes_per_element);

  // 1. Compute U1 = U +dt*V0.5
  // Update nodal displacements
  for (int j = 0; j < ndof; j++) {
    global_dof[j] = dt * vel[j];
  }

  // 2. Compute A1 = (Fext - Fint(U1)/M

  // --- Update global mass
  // TODO: Lonnie: looks like this part won't be used during initialization, is
  // it ok to perform these computations anyways?
  for (int i = 0; i < num_elements; i++) {
    // Per element variables
    for (int k = 0; k < dof_per_element; k++) {
      element_mass_matrix_diagonals[k] = 0.0;
      element_xloc[k] = 0.0;
      element_dof[k] = 0.0;
    }

    // Get the nodes of this element
    for (int j = 0; j < nodes_per_element; j++) {
      this_element_nodes[j] = element_nodes[nodes_per_element * i + j];
    }

    // Get the element locations
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_xloc, element_xloc.data());

    // Get the element degrees of freedom
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_dof, element_dof.data());

    // Calculate the element mass matrix
    Analysis::element_mass_matrix(element_density, element_xloc.data(),
                                  element_dof.data(),
                                  element_mass_matrix_diagonals.data());

    // assemble global acceleration
    for (int j = 0; j < nodes_per_element; j++) {
      int node = this_element_nodes[j];

      global_mass[3 * node] += element_mass_matrix_diagonals[3 * j];
      global_mass[3 * node + 1] += element_mass_matrix_diagonals[3 * j + 1];
      global_mass[3 * node + 2] += element_mass_matrix_diagonals[3 * j + 2];
    }
  }

  for (int i = 0; i < num_elements; i++) {
    for (int k = 0; k < dof_per_element; k++) {
      element_mass_matrix_diagonals[k] = 0.0;
      element_xloc[k] = 0.0;
      element_dof[k] = 0.0;
      element_acc[k] = 0.0;
      element_internal_forces[k] = 0.0;
    }

    for (int j = 0; j < nodes_per_element; j++) {
      this_element_nodes[j] = element_nodes[nodes_per_element * i + j];
    }

    // Get the element mass matrix
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_mass,
        element_mass_matrix_diagonals.data());

    // Get the element locations
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_xloc, element_xloc.data());

    // Get the element degrees of freedom
    Analysis::template get_element_dof<spatial_dim>(
        this_element_nodes.data(), global_dof, element_dof.data());

    T Mr_inv[dof_per_element];

    for (int k = 0; k < dof_per_element; k++) {
      Mr_inv[k] = 1.0 / element_mass_matrix_diagonals[k];
    }

    Analysis::calculate_f_internal(element_xloc.data(), element_dof.data(),
                                   element_internal_forces.data(), material);

    for (int j = 0; j < dof_per_element; j++) {
      element_acc[j] = Mr_inv[j] * (-element_internal_forces[j]);
    }

    // assemble global acceleration
    for (int j = 0; j < nodes_per_element; j++) {
      int node = this_element_nodes[j];

      global_acc[3 * node] += element_acc[3 * j];
      global_acc[3 * node + 1] += element_acc[3 * j + 1];
      global_acc[3 * node + 2] += element_acc[3 * j + 2];
    }
  }

  // Add contact forces and body forces
  for (int i = 0; i < num_nodes; i++) {
    T node_pos[3];
    node_pos[0] = global_xloc[3 * i] + global_dof[3 * i];
    node_pos[1] = global_xloc[3 * i + 1] + global_dof[3 * i + 1];
    node_pos[2] = global_xloc[3 * i + 2] + global_dof[3 * i + 2];

    T node_mass[3];
    node_mass[0] = global_mass[3 * i];
    node_mass[1] = global_mass[3 * i + 1];
    node_mass[2] = global_mass[3 * i + 2];

    // Contact Forces
    wall->detect_contact(global_acc, i, node_pos, node_mass);

    // Body Forces
    int gravity_dim = 2;
    global_acc[3 * i + gravity_dim] += -9.81;
  }
}