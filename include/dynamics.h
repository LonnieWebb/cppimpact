#pragma once
#include <iostream>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cfloat>

#include "basematerial.h"
#include "mesh.h"
#include "wall.h"

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
  Wall<T, 2, Basis> *wall;
  T *global_xloc;
  T *vel;

  Dynamics(Mesh<T> *input_mesh, BaseMaterial<T, dof_per_node> *input_material,
           Wall<T, 2, Basis> *input_wall = nullptr)
      : mesh(input_mesh),
        material(input_material),
        wall(input_wall),
        reduced_nodes(nullptr),
        reduced_dofs_size(0),
        vel(new T[mesh->num_nodes * dof_per_node]),
        global_xloc(new T[mesh->num_nodes * dof_per_node]) // Allocate memory for global_xloc here
  {
    ndof = mesh->num_nodes * dof_per_node;
  }

  ~Dynamics()
  {
    delete[] reduced_nodes;
    delete[] vel;
    delete[] global_xloc;
  }

  // Initialize the body. Move the mesh origin to init_position and give all
  // nodes init_velocity.
  void initialize(T init_position[dof_per_node],
                  T init_velocity[dof_per_node])
  {
    std::cout << "ndof: " << ndof << std::endl;
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

  void export_to_vtk(int timestep, T *vel_i)
  {
    if (timestep % 2 != 0)
      return;

    const std::string directory = "../output";
    const std::string filename = directory + "/simulation_" + std::to_string(timestep) + ".vtk";
    std::ofstream vtkFile(filename);

    if (!vtkFile.is_open())
    {
      std::cerr << "Failed to open " << filename << std::endl;
      return;
    }

    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "FEA simulation data\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";
    const double threshold = 1e6;

    vtkFile << "POINTS " << mesh->num_nodes << " float\n";
    for (int i = 0; i < mesh->num_nodes; ++i)
    {
      T x = global_xloc[3 * i];
      T y = global_xloc[3 * i + 1];
      T z = global_xloc[3 * i + 2];

      // Check for NaN or extremely large values and set to 0 if found
      if (std::isnan(x) || std::isinf(x) || std::abs(x) > threshold)
      {
        printf("Invalid value detected in x-coordinate at node %d: %f, setting to 0.\n", i, x);
        x = 0.0;
      }
      if (std::isnan(y) || std::isinf(y) || std::abs(y) > threshold)
      {
        printf("Invalid value detected in y-coordinate at node %d: %f, setting to 0.\n", i, y);
        y = 0.0;
      }
      if (std::isnan(z) || std::isinf(z) || std::abs(z) > threshold)
      {
        printf("Invalid value detected in z-coordinate at node %d: %f, setting to 0.\n", i, z);
        z = 0.0;
      }

      vtkFile << std::fixed << std::setprecision(6);
      vtkFile << x << " " << y << " " << z << "\n";
    }

    vtkFile << "CELLS " << mesh->num_elements << " " << mesh->num_elements * (nodes_per_element + 1) << "\n";
    for (int i = 0; i < mesh->num_elements; ++i)
    {
      vtkFile << nodes_per_element; // Number of points in this cell
      for (int j = 0; j < nodes_per_element; ++j)
      {
        vtkFile << " " << mesh->element_nodes[nodes_per_element * i + j];
      }
      vtkFile << "\n"; // Ensure newline at the end of each cell's connectivity list
    }

    vtkFile << "CELL_TYPES " << mesh->num_elements << "\n";
    for (int i = 0; i < mesh->num_elements; ++i)
    {
      vtkFile << "10\n"; // VTK_TETRA
    }

    vtkFile << "POINT_DATA " << mesh->num_nodes << "\n";
    vtkFile << "VECTORS velocity double\n";
    for (int i = 0; i < mesh->num_nodes; ++i)
    {
      for (int j = 0; j < 3; ++j)
      { // Check each component of the velocity
        T value = vel_i[3 * i + j];
        if (std::isnan(value))
        {
          std::cerr << "NaN detected in velocity at node " << i << ", component " << j << std::endl;
          value = 0.0;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    vtkFile.close();
    std::cout << "Exported " << filename << std::endl;
  }

  void add_element_vec_3D(const int this_element_nodes[], T *element_vec, T *global_vec)
  {
    for (int j = 0; j < nodes_per_element; j++)
    {
      int node = this_element_nodes[j];
      global_vec[3 * node] += element_vec[3 * j];
      global_vec[3 * node + 1] += element_vec[3 * j + 1];
      global_vec[3 * node + 2] += element_vec[3 * j + 2];
    }
  }

  void get_node_global_dofs(const int node_idx, int *global_dof_idx)
  {
    for (int i = 0; i < dof_per_node; i++)
    {
      global_dof_idx[i] = node_idx * dof_per_node + i;
    }
  }

  // This function is used to get the reduced degrees of freedom (DOFs) of the
  // system. It first initializes the reduced DOFs to be all DOFs, then removes
  // the fixed DOFs.
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

  void solve(double dt, double time_end)
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

    const T element_density = material->rho;
    T *global_dof = new T[ndof];
    int *element_nodes = mesh->element_nodes;

    // Copy data from mesh->xloc to global_dof
    // mesh->xloc will store initial positions
    memcpy(global_xloc, mesh->xloc, ndof * sizeof(T));
    T *next_global_dof = new T[ndof];
    // T *next_global_xloc = new T[ndof];
    // T *next_vel = new T[ndof];
    T *vel_i = new T[ndof];
    T *global_forces = new T[ndof];
    T *global_acc = new T[ndof];
    T *global_acc_accumulated = new T[ndof];
    T *global_mass = new T[ndof];
    int *accumulation_count = new int[ndof];

    for (int i = 0; i < ndof; i++)
    {
      global_dof[i] = 0.0001 * rand() / RAND_MAX;
      vel_i[i] = 0.0;
      next_global_dof[i] = 0.0;
      // next_global_xloc[i] = 0.0;
      accumulation_count[i] = 0;
      global_forces[i] = 0.0;
      global_acc_accumulated[i] = 0.0;
      global_acc[i] = 0.0;
      global_mass[i] = 0.0;
    }

    // memcpy(next_vel, vel, ndof * sizeof(T));

    T *element_mass_matrix_diagonals = new T[dof_per_element];
    T *element_xloc = new T[dof_per_element];
    T *element_dof = new T[dof_per_element];
    T *element_forces = new T[dof_per_element];
    T *element_vel = new T[dof_per_element];
    T *element_acc = new T[dof_per_element];
    T *element_internal_forces = new T[dof_per_element];
    T *element_contact_forces = new T[dof_per_element];
    T *element_Vr_i_plus_half = new T[dof_per_element];
    T *element_Vr_i_1_5 = new T[dof_per_element];
    T *element_acc_accumulated = new T[dof_per_element];

    int *this_element_nodes = new int[nodes_per_element];

    double time = 0.0;
    // T element_density;

    // Loop over all elements
    for (int i = 0; i < mesh->num_elements; i++)
    {
      for (int k = 0; k < dof_per_element; k++)
      {
        element_mass_matrix_diagonals[k] = 0.0;
        element_xloc[k] = 0.0;
        element_dof[k] = 0.0;
        element_forces[k] = 0.0;
        element_vel[k] = 0.0;
        element_acc[k] = 0.0;
        element_internal_forces[k] = 0.0;
        element_contact_forces[k] = 0.0;
        element_Vr_i_plus_half[k] = 0.0;
        element_acc_accumulated[k] = 0.0;
      }

      // element_density = element_densities[i];

      for (int j = 0; j < nodes_per_element; j++)
      {
        this_element_nodes[j] = element_nodes[nodes_per_element * i + j];
      }

      // Get the element locations
      Analysis::template get_element_dof<spatial_dim>(this_element_nodes, global_xloc,
                                                      element_xloc);
      // Get the element degrees of freedom
      Analysis::template get_element_dof<spatial_dim>(this_element_nodes, global_dof,
                                                      element_dof);
      // Get the element velocities
      Analysis::template get_element_dof<spatial_dim>(this_element_nodes, vel,
                                                      element_vel);

      Analysis::element_mass_matrix(element_density, element_xloc, element_dof,
                                    element_mass_matrix_diagonals, i);

      T Mr_inv[dof_per_element];
      for (int k = 0; k < dof_per_element; k++)
      {
        Mr_inv[k] = 1.0 / element_mass_matrix_diagonals[k];
      }

      // Initialize arrays to 0
      memset(element_internal_forces, 0, sizeof(T) * 3 * nodes_per_element);

      Analysis::calculate_f_internal(element_xloc, element_dof,
                                     element_internal_forces, material);
      memset(element_internal_forces, 0, sizeof(T) * 3 * nodes_per_element);

      // Initial Computation
      for (int j = 0; j < dof_per_element; j++)
      {
        element_acc[j] = Mr_inv[j] * (-element_internal_forces[j]);
      }

      // assemble acc
      for (int j = 0; j < nodes_per_element; j++)
      {
        int node = this_element_nodes[j];

        global_acc[3 * node] += element_acc[3 * j];
        global_acc[3 * node + 1] += element_acc[3 * j + 1];
        global_acc[3 * node + 2] += element_acc[3 * j + 2];
      }
    }

    for (int i = 0; i < mesh->num_nodes; i++)
    {
      T node_pos[3];
      node_pos[0] = global_xloc[3 * i];
      node_pos[1] = global_xloc[3 * i + 1];
      node_pos[2] = global_xloc[3 * i + 2];

      T node_mass[3];
      node_mass[0] = global_mass[3 * i];
      node_mass[1] = global_mass[3 * i + 1];
      node_mass[2] = global_mass[3 * i + 2];

      // Contact Forces
      T node_idx = i + 1;
      wall->detect_contact(global_acc, node_idx, node_pos, node_mass);

      // Body Forces
      int gravity_dim = 2;
      global_acc[3 * i + gravity_dim] += -9.81;
    }

    for (int i = 0; i < ndof; i++)
    {
      vel[i] += 0.5 * dt * global_acc[i];
    }

    //------------------- End of Initialization -------------------
    // ------------------- Start of Time Loop -------------------
    int timestep = 0;

    while (time <= time_end)
    {
      memset(global_acc, 0, sizeof(T) * ndof);
      memset(global_acc_accumulated, 0, sizeof(T) * ndof);
      memset(accumulation_count, 0, sizeof(int) * ndof);
      memset(global_dof, 0, sizeof(T) * ndof);
      memset(global_mass, 0, sizeof(T) * ndof);

      printf("Time: %f\n", time);
      for (int j = 0; j < ndof; j++)
      {
        global_dof[j] = dt * vel[j];
      }

      for (int i = 0; i < mesh->num_elements; i++)
      {

        for (int k = 0; k < dof_per_element; k++)
        {
          element_mass_matrix_diagonals[k] = 0.0;
          element_xloc[k] = 0.0;
          element_dof[k] = 0.0;
          element_forces[k] = 0.0;
          element_vel[k] = 0.0;
          element_acc[k] = 0.0;
          element_Vr_i_plus_half[k] = 0.0;
          element_acc_accumulated[k] = 0.0;
        }

        for (int j = 0; j < nodes_per_element; j++)
        {
          this_element_nodes[j] = element_nodes[nodes_per_element * i + j];
        }

        // Get the element locations
        Analysis::template get_element_dof<spatial_dim>(this_element_nodes, global_xloc,
                                                        element_xloc);
        // Get the element degrees of freedom
        Analysis::template get_element_dof<spatial_dim>(this_element_nodes, global_dof,
                                                        element_dof);

        // might be able to avoid recaculating this
        Analysis::element_mass_matrix(element_density, element_xloc, element_dof,
                                      element_mass_matrix_diagonals, i);

        for (int k = 0; k < dof_per_element; k++)
        {
          if (element_mass_matrix_diagonals[k] < 0.0)
          {
            element_mass_matrix_diagonals[k] = -element_mass_matrix_diagonals[k]; // TODO: Fix this
          }
        }

        T Mr_inv[dof_per_element];

        for (int k = 0; k < dof_per_element; k++)
        {
          Mr_inv[k] = 1.0 / element_mass_matrix_diagonals[k];
        }

        // Initialize arrays to 0
        memset(element_internal_forces, 0, sizeof(T) * 3 * nodes_per_element);

        Analysis::calculate_f_internal(element_xloc, element_dof,
                                       element_internal_forces, material);
        memset(element_internal_forces, 0, sizeof(T) * 3 * nodes_per_element);

        for (int j = 0; j < dof_per_element; j++)
        {
          element_acc[j] = Mr_inv[j] * (-element_internal_forces[j]); // TODO: check sign
        }

        // assemble acc
        for (int j = 0; j < nodes_per_element; j++)
        {
          int node = this_element_nodes[j];

          global_acc[3 * node] += element_acc[3 * j];
          global_acc[3 * node + 1] += element_acc[3 * j + 1];
          global_acc[3 * node + 2] += element_acc[3 * j + 2];

          global_mass[3 * node] += element_mass_matrix_diagonals[3 * j];
          global_mass[3 * node + 1] += element_mass_matrix_diagonals[3 * j + 1];
          global_mass[3 * node + 2] += element_mass_matrix_diagonals[3 * j + 2];
        }
      }

      for (int i = 0; i < mesh->num_nodes; i++)
      {
        T node_pos[3];
        node_pos[0] = global_xloc[3 * i] + global_dof[3 * i];
        node_pos[1] = global_xloc[3 * i + 1] + global_dof[3 * i + 1];
        node_pos[2] = global_xloc[3 * i + 2] + global_dof[3 * i + 2];

        T node_mass[3];
        node_mass[0] = global_mass[3 * i];
        node_mass[1] = global_mass[3 * i + 1];
        node_mass[2] = global_mass[3 * i + 2];

        // Contact Forces
        T node_idx = i + 1;
        wall->detect_contact(global_acc, node_idx, node_pos, node_mass);

        // Body Forces
        int gravity_dim = 2;
        global_acc[3 * i + gravity_dim] += -9.81;
      }
      printf("\n");

      for (int i = 0; i < ndof; i++)
      {
        global_xloc[i] += global_dof[i];
        vel[i] += dt * global_acc[i];

        // TODO: only run this on export steps
        vel_i[i] = vel[i] - 0.5 * dt * global_acc[i];
      }
      export_to_vtk(timestep, vel_i);
      time += dt;
      timestep += 1;
    }
  }
};
