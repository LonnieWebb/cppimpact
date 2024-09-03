#pragma once
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "basematerial.h"
#include "cppimpact_defs.h"
#include "cppimpact_utils.h"
#include "dynamics_kernels.cuh"
#include "mesh.h"
#include "tetrahedral.h"
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
  static constexpr int num_quadrature_pts = Quadrature::num_quadrature_pts;
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

  void export_to_vtk(int timestep, T *vel_i, T *acc_i, T *mass_i,
                     T *global_xloc) {
    const std::string directory = "../gpu_output";
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

  // void add_element_vec_3D(const int this_element_nodes[], T *element_vec,
  //                         T *global_vec)
  // {
  //   for (int j = 0; j < nodes_per_element; j++)
  //   {
  //     int node = this_element_nodes[j];
  //     global_vec[3 * node] += element_vec[3 * j];
  //     global_vec[3 * node + 1] += element_vec[3 * j + 1];
  //     global_vec[3 * node + 2] += element_vec[3 * j + 2];
  //   }
  // }

  void allocate() {
    // allocate global data on device
    cudaMalloc(&d_global_dof, sizeof(T) * ndof);
    cudaMalloc(&d_global_acc, sizeof(T) * ndof);
    cudaMalloc(&d_global_mass, sizeof(T) * ndof);
    cudaMalloc(&d_vel, sizeof(T) * ndof);
    cudaMalloc(&d_vel_i, sizeof(T) * ndof);
    cudaMalloc(&d_global_xloc, sizeof(T) * ndof);
    cudaMalloc(&d_element_nodes,
               sizeof(int) * nodes_per_element * mesh->num_elements);

    cudaMalloc((void **)&d_material, sizeof(decltype(*material)));
    cudaMalloc((void **)&d_wall, sizeof(decltype(*d_wall)));

    // Explicitly allocate dynamic-allocated member
    cudaMalloc((void **)&(d_wall_slave_node_indices),
               sizeof(int) * mesh->num_slave_nodes);

    cudaMemset(d_global_dof, T(0.0), sizeof(T) * ndof);
    cudaMemset(d_global_acc, T(0.0), sizeof(T) * ndof);
    cudaMemset(d_global_mass, T(0.0), sizeof(T) * ndof);
    cudaMemset(d_vel_i, T(0.0), sizeof(T) * ndof);

    cudaMemcpy(d_global_xloc, mesh->xloc, ndof * sizeof(T),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_element_nodes, mesh->element_nodes,
               sizeof(int) * nodes_per_element * mesh->num_elements,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, ndof * sizeof(T), cudaMemcpyHostToDevice);

    // Copy Material is easy for now as it doesn't contain dynamically-allocated
    // data
    cudaMemcpy(d_material, material, sizeof(decltype(*material)),
               cudaMemcpyHostToDevice);

    // Copy over POD members
    cudaMemcpy(d_wall, wall, sizeof(decltype(*wall)), cudaMemcpyHostToDevice);

    // Copy over dynamic data
    cudaMemcpy(d_wall_slave_node_indices, wall->slave_node_indices,
               sizeof(int) * mesh->num_slave_nodes, cudaMemcpyHostToDevice);

    // Point device object's data pointer to the device memory
    cudaMemcpy(&(d_wall->slave_node_indices), &d_wall_slave_node_indices,
               sizeof(int *), cudaMemcpyHostToDevice);
  }

  void deallocate() {
    cudaFree(d_global_dof);
    cudaFree(d_global_acc);
    cudaFree(d_global_mass);
    cudaFree(d_vel);
    cudaFree(d_vel_i);
    cudaFree(d_global_xloc);
    cudaFree(d_element_nodes);
    cudaFree(d_material);
    cudaFree(d_wall);
    cudaFree(d_wall_slave_node_indices);
  }

  void solve(double dt, double time_end, int export_interval) {
    // Allocate global device data
    allocate();

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

    // a. A0 = (Fext - Fint(U0))/M
    // Loop over all elements

    // TODO: this is hard-coded for now because nodes_per_element = 10 and
    // num_quadrature_pts = 5 and 64 > 50, need to properly determine this value
    // to generalize the code
    constexpr int threads_per_block = 64;
    constexpr int nodes_per_elem_num_quad =
        nodes_per_element * num_quadrature_pts;

    T *vel_i = new T[ndof];
    T *global_mass = new T[ndof];
    T *global_acc = new T[ndof];
    T time = 0.0;
    update<T, spatial_dim, nodes_per_element>
        <<<mesh->num_elements, threads_per_block>>>(
            mesh->num_elements, dt, d_material, d_wall, d_element_nodes, d_vel,
            d_global_xloc, d_global_dof, d_global_acc, d_global_mass,
            nodes_per_elem_num_quad, time);

    // Do we need this?
    cudaDeviceSynchronize();

    const int node_blocks = mesh->num_nodes / 32 + 1;
    const int ndof_blocks = ndof / 32 + 1;
    external_forces<T><<<node_blocks, 32>>>(mesh->num_nodes, d_wall,
                                            d_global_xloc, d_global_dof,
                                            d_global_mass, d_global_acc);

    // TODO: delete this
    cudaDeviceSynchronize();

    cudaMemcpy(global_acc, d_global_acc, sizeof(T) * ndof,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(global_xloc, d_global_xloc, sizeof(T) * ndof,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(global_mass, d_global_mass, sizeof(T) * ndof,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    update_velocity<T><<<ndof_blocks, 32>>>(ndof, dt, d_vel, d_global_acc);
    for (int i = 0; i < ndof; i++) {
      vel[i] += 0.5 * dt * global_acc[i];
    }

    array_to_txt<T>("gpu_vel.txt", vel, ndof);
    array_to_txt<T>("gpu_xloc.txt", global_xloc, ndof);

    int timestep = 0;
    // Time Loop

    cudaStream_t *streams;
    int num_c = 4;
    streams = new cudaStream_t[num_c];

    for (int c = 0; c < num_c; c++) {
      cudaStreamCreateWithFlags(&streams[c], cudaStreamNonBlocking);
    }

#ifdef CPPIMPACT_DEBUG_MODE
    cuda_show_kernel_error();
#endif
    // return;

    // {
    //   cudaMemcpyAsync(vel_i, d_vel, ndof * sizeof(T), cudaMemcpyDeviceToHost,
    //                   streams[0]);
    //   cudaMemcpyAsync(global_acc, d_global_acc, ndof * sizeof(T),
    //                   cudaMemcpyDeviceToHost, streams[1]);
    //   cudaMemcpyAsync(global_mass, d_global_mass, ndof * sizeof(T),
    //                   cudaMemcpyDeviceToHost, streams[2]);
    //   cudaMemcpyAsync(global_xloc, d_global_xloc, ndof * sizeof(T),
    //                   cudaMemcpyDeviceToHost, streams[3]);

    //   cudaStreamSynchronize(streams[0]);
    //   cudaStreamSynchronize(streams[1]);
    //   cudaStreamSynchronize(streams[2]);
    //   cudaStreamSynchronize(streams[3]);
    //   export_to_vtk(timestep, vel_i, global_acc, global_mass, global_xloc);
    // };

    while (time <= time_end) {
      cudaMemsetAsync(d_global_acc, T(0.0), sizeof(T) * ndof, streams[0]);
      cudaMemsetAsync(d_global_dof, T(0.0), sizeof(T) * ndof, streams[1]);
      // cudaMemsetAsync(d_global_mass, T(0.0), sizeof(T) * ndof, streams[2]);
      cudaStreamSynchronize(streams[0]);
      cudaStreamSynchronize(streams[1]);
      cudaStreamSynchronize(streams[2]);
      printf("Time: %f\n", time);

      update_dof<T>
          <<<ndof_blocks, 32, 0, streams[0]>>>(ndof, dt, d_vel, d_global_dof);
      cudaStreamSynchronize(streams[0]);

      update<T, spatial_dim, nodes_per_element>
          <<<mesh->num_elements, threads_per_block, 0, streams[0]>>>(
              mesh->num_elements, dt, d_material, d_wall, d_element_nodes,
              d_vel, d_global_xloc, d_global_dof, d_global_acc, d_global_mass,
              nodes_per_elem_num_quad, time);
      cudaStreamSynchronize(streams[0]);

      external_forces<T><<<node_blocks, 32, 0, streams[0]>>>(
          mesh->num_nodes, d_wall, d_global_xloc, d_global_dof, d_global_mass,
          d_global_acc);
      cudaStreamSynchronize(streams[0]);

      timeloop_update<T><<<ndof_blocks, 32, 0, streams[0]>>>(
          ndof, dt, d_global_xloc, d_vel, d_global_acc, d_vel_i, d_global_dof);
      cudaStreamSynchronize(streams[0]);

      // TODO: exporting
      if (timestep % export_interval == 0) {
        cudaMemcpyAsync(vel_i, d_vel, ndof * sizeof(T), cudaMemcpyDeviceToHost,
                        streams[0]);
        cudaMemcpyAsync(global_acc, d_global_acc, ndof * sizeof(T),
                        cudaMemcpyDeviceToHost, streams[1]);
        cudaMemcpyAsync(global_mass, d_global_mass, ndof * sizeof(T),
                        cudaMemcpyDeviceToHost, streams[2]);
        cudaMemcpyAsync(global_xloc, d_global_xloc, ndof * sizeof(T),
                        cudaMemcpyDeviceToHost, streams[3]);

        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        cudaStreamSynchronize(streams[2]);
        cudaStreamSynchronize(streams[3]);
        export_to_vtk(timestep, vel_i, global_acc, global_mass, global_xloc);
      };

      time += dt;
      timestep += 1;

#ifdef CPPIMPACT_DEBUG_MODE
      cuda_show_kernel_error();
#endif
    }

    for (int c = 0; c < num_c; c++) {
      cudaStreamDestroy(streams[c]);
    }
    delete[] streams;

    deallocate();
  }

 private:
  // Host data pointers

  // Device data pointers
  T *d_global_dof = nullptr;
  T *d_global_acc = nullptr;
  T *d_global_mass = nullptr;
  T *d_vel = nullptr;
  T *d_vel_i = nullptr;
  T *d_global_xloc = nullptr;

  int *d_element_nodes = nullptr;

  BaseMaterial<T, spatial_dim> *d_material = nullptr;
  Wall<T, 2, Basis> *d_wall = nullptr;
  int *d_wall_slave_node_indices = nullptr;
};
