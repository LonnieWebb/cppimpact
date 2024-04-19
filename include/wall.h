#pragma once
#include <string>

template <typename T, int dim, class Basis>
class Wall {
 public:
  // dim denotes the plane which the wall is parallel with (0, 1, 2) for (x, y,
  // z) respectively. normal denotes which way direction the surface of the wall
  // faces 1 for positive x/y/z, -1 for negative x/y/z

  std::string name;
  T location;
  T stiffness;
  int *slave_node_indices;
  int num_slave_nodes;
  int normal;
  // static_assert(normal == 1 || normal == -1, "Normal must be either 1 or
  // -1.");

  Wall(std::string name, T location, T stiffness, int *slave_node_indices,
       int num_slave_nodes, int normal)
      : name(name),
        location(location),
        stiffness(stiffness),
        slave_node_indices(slave_node_indices),
        num_slave_nodes(num_slave_nodes),
        normal(normal) {
    // for (int i = 0; i < num_slave_nodes; i++) {
    //   std::cout << "slave_node_indices[i]: " << slave_node_indices[i]
    //             << std::endl;
    // }
  }

  ~Wall() {}

  void detect_contact(T *global_acc, int node_idx, T *node_pos, T *node_mass) {
    for (int j = 0; j < num_slave_nodes; j++) {
      if (node_idx == slave_node_indices[j]) {
        T wall_distance = (node_pos[dim] - location) * normal;
        if (wall_distance < 0.0) {
          global_acc[3 * (node_idx - 1) + dim] +=
              -1 * (1 / node_mass[dim]) * stiffness * wall_distance * normal;
        }
      }
    }
  }
};