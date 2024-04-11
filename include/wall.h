#pragma once
#include <string>

template <typename T, int dim, class Basis>
class Wall
{
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
  // static_assert(normal == 1 || normal == -1, "Normal must be either 1 or -1.");

  Wall(std::string name, T location, T stiffness, int *slave_node_indices,
       int num_slave_nodes, int normal)
      : name(name),
        location(location),
        stiffness(stiffness),
        slave_node_indices(slave_node_indices),
        num_slave_nodes(num_slave_nodes),
        normal(normal)
  {
    // for (int i = 0; i < num_slave_nodes; i++) {
    //   std::cout << "slave_node_indices[i]: " << slave_node_indices[i]
    //             << std::endl;
    // }
  }

  ~Wall() {}

  void detect_contact(T *element_xloc, int *this_element_nodes, T *contact_forces)
  {
    for (size_t i = 0; i < Basis::nodes_per_element; i++)
    {
      for (int j = 0; j < num_slave_nodes; j++)
      {
        if (this_element_nodes[i] == slave_node_indices[j])
        {

          T wall_distance = (element_xloc[3 * i + dim] - location) * normal;
          if (wall_distance < 0.0)
          {
            printf("Contact detected at node %i with penetration %f \n",
                   slave_node_indices[j], -wall_distance);
            contact_forces[3 * i + dim] += stiffness * wall_distance * normal;
          }
        }
      }
    }
  }
};