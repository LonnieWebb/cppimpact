#include <string>
#include <cuda_runtime.h>

#include "include/analysis.h"
#include "include/mesh.h"
#include "include/physics.h"
#include "include/tetrahedral.h"
#include "include/dynamics.h"
#include "include/elastoplastic.h"
#include "include/wall.h"
#include <cblas.h>

int main(int argc, char *argv[])
{
    using T = double;
    using Basis = TetrahedralBasis<T>;
    using Quadrature = TetrahedralQuadrature;
    using Physics = NeohookeanPhysics<T>;
    using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

    const int dof_per_node = 3;

    std::vector<std::string> node_set_names;
    // Load in the mesh
    std::string filename("../input/Dynamics Cube.inp");
    Mesh<T> tensile;

    // Material Properties
    T E = 200E9;  // Pa
    T rho = 7800; // kg/m3
    T nu = 0.25;
    T beta = 0.0;
    T H = 10;
    T Y0 = 1.9 * std::sqrt(3.0);
    std::string name = "Steel";

    Elastoplastic<T, dof_per_node> material(E, rho, nu, beta, H, Y0, name);
    tensile.load_mesh(filename);

    // Set the number of degrees of freedom

    // Position and velocity in x, y, z
    T init_position[] = {0.0, 0.0, 0.0};
    T init_velocity[] = {0, 0.0, -10};

    const int normal = 1;
    std::string wall_name = "Wall";
    T location = -0.3;
    double dt = 0.002;
    double time_end = 2;

    Wall<T, 2, Basis> w(wall_name, location, E, tensile.slave_nodes, tensile.num_slave_nodes, normal);

    Dynamics<T, Basis, Analysis> dyna(&tensile, &material, &w);
    dyna.initialize(init_position, init_velocity);

    dyna.solve(dt, time_end);

    return 0;
}