#pragma once
#include <stdexcept>

#include "basematerial.h"

template <typename T, int dof_per_node>
class Elastoplastic : public BaseMaterial<T, dof_per_node> {
 public:
  Elastoplastic(T E, T rho, T nu, T beta, T H, T Y0,
                std::string name = "Material")
      : BaseMaterial<T, dof_per_node>("InfinitesimalElastoplastic", E, rho, nu,
                                      name),
        beta(beta),
        H(H),
        Y0_initial_yield_stress(Y0) {}

  virtual ~Elastoplastic() {}

  // Implement pure virtual functions from BaseMaterial
  virtual void D_compute_stiffnessMatrix3D(T* output) const override{};
  //   virtual void D_compute_stiffnessMatrix2D(T* output) const override;
  virtual void compute_dstrain(const T* gradient_u, int num_integrations,
                               T* output) const override{
      /*Compute the strain given the gradient of displacement du / dx.For small
         strain, doesnt matter if it is Right Cauchy Green or left strain
         tensor*/
      // T* eye[ndof * ndof];
      // int eye_r = ndof;
      // int eye_c = ndof;
      // // populate eye
      // for (int i = 0; i < ndof; ++i) {
      //   for (int j = 0; j < ndof; ++j) {
      //     if (i == j) {
      //       eye[i * ndof + j] = 1;  // Diagonal elements set to 1
      //     } else {
      //       eye[i * ndof + j] = 0;  // Off-diagonal elements set to 0
      //     }
      //   }
      // }

      // if (ndof == 3) {
      //   T dstrains[num_integrations * 6];
      //   int dstrains_r = num_integrations;
      //   int dstrains_c = 6;

      //   for (int lx = 0; lx < num_integrations; lx++) {
      //     gradient_u = gradu[lx];
      //     // TODO: Complete this
      //   }

      // } else if (ndof == 2) {
      //   throw std::logic_error("2D strain calc not implemented");
      // }
  };
  virtual void compute_stress(const T* dstrain, const T* strain,
                              T* output) const override{};
  virtual void compute_wave_speed3D(T* output) override{};
  virtual void compute_wave_speed2D(T* output) override{};

 protected:
  T beta;
  T H;
  T Y0_initial_yield_stress;
  // Additional member variables for elastoplastic-specific properties
};