#pragma once

#include <string>

template <typename T, int dof_per_node>
class BaseMaterial
{
public:
  int ndof;
  std::string name;
  std::string type;
  T E;
  T rho;
  T nu;

  BaseMaterial(std::string type, T E, T rho, T nu,
               std::string name = "Material")
      : name(name),
        type(type),
        E(E),
        rho(rho),
        nu(nu) {}

  virtual ~BaseMaterial() {}

  // Pure virtual functions for the derived classes to implement
  virtual void D_compute_stiffnessMatrix3D(T *output) const = 0;
  //   virtual void D_compute_stiffnessMatrix2D(T* output) const = 0;
  virtual void compute_dstrain(const T *gradient_u, int num_integrations,
                               T *output) const = 0;
  virtual void compute_stress(const T *dstrain, const T *strain,
                              T *output) const = 0;
  virtual void compute_wave_speed3D(T *output) = 0;
  virtual void compute_wave_speed2D(T *output) = 0;
};
