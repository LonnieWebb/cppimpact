#pragma once
#include <string>

template <typename T, int dim, int normal>
class Wall {
 public:
  // dim denotes the plane which the wall is parallel with (0, 1, 2) for (x, y,
  // z) respectively. normal denotes which way direction the surface of the wall
  // faces 1 for positive x/y/z, -1 for negative x/y/z

  std::string name;
  T location;
  T stiffness;
  int* slave_node_indices;
  int num_slave_nodes;
  static_assert(normal == 1 || normal == -1, "Normal must be either 1 or -1.");

  Wall(std::string name, T* location, T stiffness, int* slave_node_indices)
      : name(name),
        location(location),
        stiffness(stiffness),
        slave_node_indices(slave_node_indices) {}

  ~Wall() {}

  void detect_contact(T* xloc) {
    for (int i = 0; i < num_slave_nodes; i++) {
      T wall_distance = (xloc[3 * i + dim] - location) * normal;
      if (wall_distance < 1e-5) {
        printf("Contact detected at node %i with penetration %f",
               slave_node_indices[i], wall_distance);
        Fcontact = stiffness * wall_distance * normal;
        // TODO: add_contact_force
      }
    }
  }
};