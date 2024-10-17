#pragma once

#include "cppimpact_defs.h"
#include "elastoplastic.h"
#include "physics.h"
#include "tetrahedral.h"

// Forward declaration of the FEAnalysis template
template <typename T, class Basis, class Quadrature, class Physics>
class FEAnalysis;

// Define the scalar type
using T = double;

// Basis Type
#if defined(USE_LINEAR_BASIS)
using Basis = TetrahedralBasisLinear<T>;

#elif defined(USE_QUADRATIC_BASIS)
using Basis = TetrahedralBasisQuadratic<T>;

#else
using Basis = TetrahedralBasisLinear<T>;
#endif

// Quadrature Type
using Quadrature = TetrahedralQuadrature5pts;

// Physics Type
using Physics = NeohookeanPhysics<T>;

// Analysis Type
using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

// Degrees of Freedom per Node
constexpr int dof_per_node = 3;
