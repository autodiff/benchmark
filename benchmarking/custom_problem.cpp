#include <Eigen/Core>

// clang-format off
#include "wrappers/adolc.hpp"
#include "wrappers/autodiff_dual.hpp"
/// autodiff forward (var) can not be used at the same time
// #include "wrappers/autodiff_var.hpp"
#include "wrappers/autodiff_real.hpp"
#include "wrappers/adept.hpp"
#include "wrappers/autodiff_dual.hpp"
#include "wrappers/ceres.hpp"
#include "wrappers/cppadcg.hpp"
#include "wrappers/cppad.hpp"
#include "wrappers/numerical.hpp"
#include "wrappers/sacado.hpp"
// clang-format on

#include "testing.hpp"

struct CustomTest
{
  /// Problem input size at compile time (set to -1 for dynamic)
  static constexpr int InputSize = -1;

  CustomTest(int n) : n_(n) {}

  /// Test name
  std::string name() const
  {
    return std::string("CustomTest_") + std::to_string(n_);
  }

  /// Function that returns dynamic input size
  int input_size() const { return n_; }

  /// Function that is to be benchmarked
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 1, 1> operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    Eigen::Matrix<typename Derived::Scalar, 1, 1> ret;
    ret(0) = x.squaredNorm();
    return ret;
  }

  int n_;
};

int main()
{
  std::vector<CustomTest> tests;
  tests.emplace_back(5);
  tests.emplace_back(10);
  tests.emplace_back(15);

  for (const auto & test : tests) {
    ad_testing::run_speedtest<ad_testing::AdolcWrapper>(test);
    ad_testing::run_speedtest<ad_testing::AdolcTapelessWrapper>(test);
    ad_testing::run_speedtest<ad_testing::AdeptWrapper>(test);
    ad_testing::run_speedtest<ad_testing::AutodiffDualWrapper>(test);
    ad_testing::run_speedtest<ad_testing::AutodiffRealWrapper>(test);
    // Ceres only works for static-sized tests
    // ad_testing::run_speedtest<ad_testing::CeresWrapper>(test);
    ad_testing::run_speedtest<ad_testing::CppADCGWrapper>(test);
    ad_testing::run_speedtest<ad_testing::CppADWrapper>(test);
    ad_testing::run_speedtest<ad_testing::NumericalWrapper>(test);
    ad_testing::run_speedtest<ad_testing::SacadoWrapper>(test);
  }

  return 0;
}
