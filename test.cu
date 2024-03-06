#include <string>
#include <cuda_runtime.h>

#include "include/analysis.h"
#include "include/mesh.h"
#include "include/physics.h"
#include "include/tetrahedral.h"

int main(int argc, char *argv[])
{
    using T = double;
    using Basis = TetrahedralBasis;
    using Quadrature = TetrahedralQuadrature;
    using Physics = NeohookeanPhysics<T>;
    using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

    std::vector<std::string> node_set_names;

    // Load in the mesh
    std::string filename("../input/Tensile1.inp");
    Mesh<T> tensile;
    tensile.load_mesh(filename);

    // Set the number of degrees of freedom
    int ndof = 3 * tensile.num_nodes;
    for (int i = 0; i < tensile.num_fixed_nodes; i++)
    {
        std::cout << tensile.fixed_nodes[i] << std::endl;
    }
}