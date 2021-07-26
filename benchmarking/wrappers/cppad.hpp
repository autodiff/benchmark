#ifndef WRAPPERS__CPPAD_HPP_
#define WRAPPERS__CPPAD_HPP_

#include <cppad/cppad.hpp>
#include <Eigen/Core>

namespace ad_testing {

class CppADWrapper
{
public:
  static constexpr char name[] = "CppAD";

  template<typename Func, typename Derived>
  void setup(Func && f, const Eigen::PlainObjectBase<Derived> & x)
  {
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ax = x.template cast<CppAD::AD<double>>();

    CppAD::Independent(ax);
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ay = f(ax);
    f_ad                                                   = CppAD::ADFun<double>(ax, ay);

    f_ad.optimize();
  }

  template<typename Func, typename Derived>
  void run(Func &&, const Eigen::PlainObjectBase<Derived> & x, Eigen::MatrixXd & J)
  {
    Eigen::VectorXd x_dyn = x;
    auto j_ad             = f_ad.Jacobian(x_dyn);
    J                     = Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
      j_ad.data(), f_ad.Range(), f_ad.Domain());
  }

private:
  CppAD::ADFun<double> f_ad;
};

}  // namespace ad_testing

#endif  // WRAPPERS__CPPAD_HPP_
