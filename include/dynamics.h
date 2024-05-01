#pragma once
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "basematerial.h"
#include "cppimpact_utils.h"
#include "dynamics_kernels.h"
#include "mesh.h"
#include "wall.h"

template <typename T, class Basis, class Analysis, class Quadrature>
class Dynamics {
 public:
  Mesh<T> *mesh;

  int *reduced_nodes;
  int reduced_dofs_size;
  int ndof;
  static constexpr int nodes_per_element = Basis::nodes_per_element;
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int dof_per_node = spatial_dim;
  static constexpr int num_quadrature_pts = Quadrature::num_quadrature_pts;
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
        global_xloc(
            new T[mesh->num_nodes *
                  dof_per_node])  // Allocate memory for global_xloc here
  {
    ndof = mesh->num_nodes * dof_per_node;
  }

  ~Dynamics() {
    delete[] reduced_nodes;
    delete[] vel;
    delete[] global_xloc;
  }

  // Initialize the body. Move the mesh origin to init_position and give all
  // nodes init_velocity.
  void initialize(T init_position[dof_per_node],
                  T init_velocity[dof_per_node]) {
    std::cout << "ndof: " << ndof << std::endl;
    for (int i = 0; i < mesh->num_nodes; i++) {
      vel[3 * i] = init_velocity[0];
      vel[3 * i + 1] = init_velocity[1];
      vel[3 * i + 2] = init_velocity[2];

      mesh->xloc[3 * i] = mesh->xloc[3 * i] + init_position[0];
      mesh->xloc[3 * i + 1] = mesh->xloc[3 * i + 1] + init_position[1];
      mesh->xloc[3 * i + 2] = mesh->xloc[3 * i + 2] + init_position[2];
    }
  }

  void export_to_vtk(int timestep, T *vel_i, T *acc_i, T *mass_i) {
    const std::string directory = "../cpu_output";
    const std::string filename =
        directory + "/simulation_" + std::to_string(timestep) + ".vtk";
    std::ofstream vtkFile(filename);

    if (!vtkFile.is_open()) {
      std::cerr << "Failed to open " << filename << std::endl;
      return;
    }

    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "FEA simulation data\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET UNSTRUCTURED_GRID\n";
    const double threshold = 1e6;

    vtkFile << "POINTS " << mesh->num_nodes << " float\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      T x = global_xloc[3 * i];
      T y = global_xloc[3 * i + 1];
      T z = global_xloc[3 * i + 2];

      // Check for NaN or extremely large values and set to 0 if found
      if (std::isnan(x) || std::isinf(x) || std::abs(x) > threshold) {
        printf(
            "Invalid value detected in x-coordinate at node %d: %f, setting to "
            "0.\n",
            i, x);
        x = 0.0;
      }
      if (std::isnan(y) || std::isinf(y) || std::abs(y) > threshold) {
        printf(
            "Invalid value detected in y-coordinate at node %d: %f, setting to "
            "0.\n",
            i, y);
        y = 0.0;
      }
      if (std::isnan(z) || std::isinf(z) || std::abs(z) > threshold) {
        printf(
            "Invalid value detected in z-coordinate at node %d: %f, setting to "
            "0.\n",
            i, z);
        z = 0.0;
      }

      vtkFile << std::fixed << std::setprecision(6);
      vtkFile << x << " " << y << " " << z << "\n";
    }

    vtkFile << "CELLS " << mesh->num_elements << " "
            << mesh->num_elements * (nodes_per_element + 1) << "\n";
    for (int i = 0; i < mesh->num_elements; ++i) {
      vtkFile << nodes_per_element;  // Number of points in this cell
      for (int j = 0; j < nodes_per_element; ++j) {
        vtkFile << " " << mesh->element_nodes[nodes_per_element * i + j];
      }
      vtkFile << "\n";  // Ensure newline at the end of each cell's connectivity
                        // list
    }

    vtkFile << "CELL_TYPES " << mesh->num_elements << "\n";
    for (int i = 0; i < mesh->num_elements; ++i) {
      vtkFile << "10\n";  // VTK_TETRA
    }

    vtkFile << "POINT_DATA " << mesh->num_nodes << "\n";
    vtkFile << "VECTORS velocity double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {  // Check each component of the velocity
        T value = vel_i[3 * i + j];
        if (std::isnan(value)) {
          std::cerr << "NaN detected in velocity at node " << i
                    << ", component " << j << std::endl;
          value = 0.0;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    vtkFile << "VECTORS acceleration double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {
        T value = acc_i[3 * i + j];
        if (std::isnan(value)) {
          std::cerr << "NaN detected in acceleration at node " << i
                    << ", component " << j << std::endl;
          value = 0.0;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    vtkFile << "VECTORS mass double\n";
    for (int i = 0; i < mesh->num_nodes; ++i) {
      for (int j = 0; j < 3; ++j) {
        T value = mass_i[3 * i + j];
        if (std::isnan(value) || value < 0.0) {
          std::cerr << "Invalid value detected in mass at node " << i
                    << ", component " << j << std::endl;
        }
        vtkFile << value << (j < 2 ? " " : "\n");
      }
    }

    vtkFile.close();
    std::cout << "Exported " << filename << std::endl;
  }

  void solve(double dt, double time_end, int export_interval) {
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
    printf("Solving dynamics\n");

    // Material and mesh information
    int *element_nodes = mesh->element_nodes;

    // Allocate global data
    T *global_dof = new T[ndof];
    T *global_acc = new T[ndof];
    T *global_mass = new T[ndof];
    // Initialize global data
    memcpy(global_xloc, mesh->xloc,
           ndof * sizeof(T));  // mesh->xloc will store initial positions
    for (int i = 0; i < ndof; i++) {
      global_dof[i] = 0.0;
      global_acc[i] = 0.0;
      global_mass[i] = 0.0;
    }

    // Intermediate velocity for vtk export
    T *vel_i = new T[ndof];
    for (int i = 0; i < ndof; i++) {
      vel_i[i] = 0.0;
    }

    // Initialize states
    update<T, spatial_dim, nodes_per_element, Basis, Analysis>(
        mesh->num_nodes, mesh->num_elements, ndof, dt, material, wall,
        element_nodes, vel, global_xloc, global_dof, global_acc, global_mass);

    // b.Stagger V0 .5 = V0 + dt / 2 * a0
    // Update velocity
    for (int i = 0; i < ndof; i++) {
      vel[i] += 0.5 * dt * global_acc[i];
    }

    array_to_txt<T>("cpu_vel.txt", vel, ndof);

    //------------------- End of Initialization -------------------
    // ------------------- Start of Time Loop -------------------
    double time = 0.0;
    int timestep = 0;

    while (time <= time_end) {
      printf("Time: %f\n", time);

      memset(global_dof, 0, sizeof(T) * ndof);
      // 1. Compute U1 = U +dt*V0.5
      // Update nodal displacements
      for (int j = 0; j < ndof; j++) {
        global_dof[j] = dt * vel[j];
      }

      update<T, spatial_dim, nodes_per_element, Basis, Analysis>(
          mesh->num_nodes, mesh->num_elements, ndof, dt, material, wall,
          element_nodes, vel, global_xloc, global_dof, global_acc, global_mass);

      // Compute total mass (useful?)
      T total_mass = 0.0;
      for (int i = 0; i < ndof; i++) {
        total_mass += global_mass[i] / 3.0;
      }
      // printf("mass: %30.15e\n", total_mass);

      // 3. Compute V1.5 = V0.5 + A1*dt
      // 3. Compute V1 = V1.5 - dt/2 * a1
      // 4. Loop back to 1.

      for (int i = 0; i < ndof; i++) {
        global_xloc[i] += global_dof[i];
        vel[i] += dt * global_acc[i];

        // TODO: only run this on export steps
        vel_i[i] = vel[i] - 0.5 * dt * global_acc[i];
      }

      if (timestep % export_interval == 0) {
        export_to_vtk(timestep, vel_i, global_acc, global_mass);
      }
      time += dt;
      timestep += 1;
    }
  }
};
