#include <string>
#include <cuda_runtime.h>

#include "analysis.h"
#include "mesh.h"
#include "physics.h"
#include "tetrahedral.h"

int main(int argc, char *argv[])
{
    using T = double;
    using Basis = TetrahedralBasis;
    using Quadrature = TetrahedralQuadrature;
    using Physics = NeohookeanPhysics<T>;
    using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

    int num_elements, num_nodes, num_node_sets;
    int *element_nodes, *node_set_starts, *node_set_indices;
    T *xloc;
    std::vector<std::string> node_set_names;

    // Load in the mesh
    std::string filename("../input/Tensile1.inp");
    load_mesh<T>(filename, &num_elements, &num_nodes,
                 &num_node_sets, &element_nodes, &xloc,
                 &node_set_starts, &node_set_indices,
                 &node_set_names);

    // Set the number of degrees of freedom
    int ndof = 3 * num_nodes;
}