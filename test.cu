#include <string>
#include <cuda_runtime.h>

#include "include/analysis.h"
#include "include/mesh.h"
#include "include/physics.h"
#include "include/tetrahedral.h"
#include "include/dynamics.h"
#include "include/elastoplastic.h"
#include "include/wall.h"

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
    std::string filename("../input/Tensile1.inp");
    Mesh<T> tensile;

    // Material Properties
    T E = 270;
    T rho = 1.0;
    T nu = 0.3;
    T beta = 0.0;
    T H = 10;
    T Y0 = 1.9 * std::sqrt(3.0);
    std::string name = "Steel";

    Elastoplastic<T, dof_per_node> material(E, rho, nu, beta, H, Y0, name);
    tensile.load_mesh(filename);

    // Set the number of degrees of freedom

    Dynamics<T, Basis, Analysis> dyna(&tensile, &material);

    // Position and velocity in x, y, z
    T init_position[] = {0.0, 1.0, 0.0};
    T init_velocity[] = {0.0, -0.1, 0.0};

    dyna.initialize(init_position, init_velocity);

    const int normal = -1;
    std::string wall_name = "Wall";
    T location = 3.0;
    double dt = 0.0012;

    Wall<T, dof_per_node, normal> w(wall_name, location, E, tensile.slave_nodes, tensile.num_slave_nodes);

    dyna.solve(dt);

    return 0;
}