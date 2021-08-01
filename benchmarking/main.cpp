#include <Eigen/Core>

#include <iomanip>
#include <iostream>
#include <tuple>

#include "wrappers/numerical.hpp"
#ifdef ENABLE_ADEPT
#include "wrappers/adept.hpp"
#endif
#ifdef ENABLE_ADOLC
#include "wrappers/adolc.hpp"
#endif
#ifdef ENABLE_AUTODIFF_DUAL
#include "wrappers/autodiff_dual.hpp"
#endif
#ifdef ENABLE_AUTODIFF_REAL
#include "wrappers/autodiff_real.hpp"
#endif
#ifdef ENABLE_AUTODIFF_VAR
#include "wrappers/autodiff_var.hpp"
#endif
#ifdef ENABLE_CERES
#include "wrappers/ceres.hpp"
#endif
#ifdef ENABLE_CERES
#include "wrappers/ceres.hpp"
#endif
#ifdef ENABLE_CPPADCG
#include "wrappers/cppadcg.hpp"  // must be before cppad
#endif
#ifdef ENABLE_CPPAD
#include "wrappers/cppad.hpp"
#endif
#ifdef ENABLE_SACADO
#include "wrappers/sacado.hpp"
#endif

#include "testing.hpp"
#include "tests.hpp"

using namespace ad_testing;

int main()
{
  auto basic_tests = std::make_tuple(Constant<4>(4),
    Constant<-1>(4),
    Constant<10>(10),
    Constant<-1>(10),
    Constant<-1>(100),
    Constant<-1>(1000),
    ManyToOne<4>(4),
    ManyToOne<-1>(4),
    ManyToOne<10>(10),
    ManyToOne<-1>(10),
    ManyToOne<-1>(100),
    ManyToOne<-1>(1000),
    OneToMany<4>(4),
    OneToMany<-1>(4),
    OneToMany<10>(10),
    OneToMany<-1>(10),
    OneToMany<-1>(100),
    OneToMany<-1>(1000),
    NeuralNet<4>(4),
    NeuralNet<-1>(4),
    NeuralNet<10>(10),
    NeuralNet<-1>(10),
    NeuralNet<-1>(100),
    NeuralNet<-1>(1000));

  auto ode_tests = std::make_tuple(ODE<4>(4), ODE<10>(10), ODE<-1>(40));

  auto manipulator_tests = std::make_tuple(
    Manipulator<4>(4), Manipulator<10>(10), Manipulator<-1>(100), Manipulator<-1>(1000));

  auto reprojection_tests = std::make_tuple(ReprojectionError<4>(4),
    ReprojectionError<10>(10),
    ReprojectionError<-1>(100),
    ReprojectionError<-1>(1000));

  auto se3_ode_tests =
    std::make_tuple(SE3ODE<4>(4), SE3ODE<10>(10), SE3ODE<100>(100), SE3ODE<1000>(1000));

  // these can run all tests (but do not necessarily succeed)
#ifdef ENABLE_ADEPT
  run_tests<AdeptWrapper>(basic_tests);
  run_tests<AdeptWrapper>(ode_tests);
  run_tests<AdeptWrapper>(manipulator_tests);
  run_tests<AdeptWrapper>(reprojection_tests);
  run_tests<AdeptWrapper>(se3_ode_tests);
#endif
#ifdef ENABLE_ADOLC
  run_tests<AdolcTapelessWrapper>(basic_tests);
  run_tests<AdolcTapelessWrapper>(ode_tests);
  run_tests<AdolcTapelessWrapper>(manipulator_tests);
  run_tests<AdolcTapelessWrapper>(reprojection_tests);
  run_tests<AdolcTapelessWrapper>(se3_ode_tests);
  // Adolc taped can not handle Eigen quaterion-vector product
  run_tests<AdolcWrapper>(basic_tests);
  run_tests<AdolcWrapper>(ode_tests);
#endif
#ifdef ENABLE_AUTODIFF_DUAL
  run_tests<AutodiffDualWrapper>(basic_tests);
  run_tests<AutodiffDualWrapper>(ode_tests);
  run_tests<AutodiffDualWrapper>(manipulator_tests);
  run_tests<AutodiffDualWrapper>(reprojection_tests);
  run_tests<AutodiffDualWrapper>(se3_ode_tests);
#endif
#ifdef ENABLE_AUTODIFF_REAL
  // Real can not handle atan2
  run_tests<AutodiffRealWrapper>(basic_tests);
  run_tests<AutodiffRealWrapper>(ode_tests);
#endif
#ifdef ENABLE_AUTODIFF_VAR
  // * AutodiffRev times out on ODE and large Manipulator tests
  run_tests<AutodiffVarWrapper>(basic_tests);
  run_tests<AutodiffVarWrapper>(std::make_tuple(Manipulator<4>(4), Manipulator<10>(10)));
  run_tests<AutodiffVarWrapper>(reprojection_tests);
#endif
#ifdef ENABLE_CPPAD
  run_tests<CppADWrapper>(basic_tests);
  run_tests<CppADWrapper>(ode_tests);
  run_tests<CppADWrapper>(manipulator_tests);
  run_tests<CppADWrapper>(reprojection_tests);
  run_tests<CppADWrapper>(se3_ode_tests);
#endif
#ifdef ENABLE_CPPADCG
  run_tests<CppADCGWrapper>(basic_tests);
  run_tests<CppADCGWrapper>(ode_tests);
  run_tests<CppADCGWrapper>(manipulator_tests);
  run_tests<CppADCGWrapper>(reprojection_tests);
  run_tests<CppADCGWrapper>(se3_ode_tests);
#endif
#ifdef ENABLE_NUMERICAL
  run_tests<NumericalWrapper>(basic_tests);
  run_tests<NumericalWrapper>(ode_tests);
  run_tests<NumericalWrapper>(manipulator_tests);
  run_tests<NumericalWrapper>(reprojection_tests);
  run_tests<NumericalWrapper>(se3_ode_tests);
#endif
#ifdef ENABLE_SACADO
  run_tests<SacadoWrapper>(basic_tests);
  run_tests<SacadoWrapper>(ode_tests);
  run_tests<SacadoWrapper>(manipulator_tests);
  run_tests<SacadoWrapper>(reprojection_tests);
  run_tests<SacadoWrapper>(se3_ode_tests);
#endif
#ifdef ENABLE_CERES
  // Ceres can't handle ODE tests because no conversion double -> Jet
  // Also can't handle dynamically sized tests
  auto ceres_tests = std::make_tuple(Constant<4>(4),
    Constant<4>(4),
    Constant<10>(10),
    ManyToOne<4>(4),
    ManyToOne<10>(10),
    OneToMany<4>(4),
    OneToMany<10>(10),
    NeuralNet<4>(4),
    NeuralNet<10>(10),
    Manipulator<4>(4),
    Manipulator<10>(10),
    ReprojectionError<10>(10));
  run_tests<CeresWrapper>(ceres_tests);
#endif

  return 0;
}
