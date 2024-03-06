#pragma once
#include <iostream>

template <typename T>
class Dynamics {
 public:
  int num_elements, num_nodes, num_node_sets;
  int *element_nodes, *node_set_starts, *node_set_indices;
  T* xloc;
  std::vector<std::string> node_set_names;

  void get_reduced_dofs() {
    // Extract the free dofs. The free dofs can be accessed via reduced_dofs
    // variable.
  }

  void update_mesh(int new_num_elements, int new_num_nodes,
                   int* new_element_nodes, T* new_xloc) {
    num_elements = new_num_elements;
    num_nodes = new_num_nodes;
    element_nodes = new_element_nodes;
    xloc = new_xloc;
  }

  void solve() {
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
  }
};