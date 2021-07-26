#ifndef WRAPPERS__NUMERICAL_HPP_
#define WRAPPERS__NUMERICAL_HPP_

#include <unsupported/Eigen/NumericalDiff>
#include <utility>

#include "common.hpp"

namespace ad_testing {

class NumericalWrapper
{
public:
  static constexpr char name[] = "Numerical";

  template<typename Func, typename Derived>
  void setup(Func && f, const Eigen::PlainObjectBase<Derived> & x)
  {
    ny_ = f(x).size();
  }

  template<typename Func, typename Derived>
  void run(Func && f, const Eigen::PlainObjectBase<Derived> & x, Eigen::MatrixXd & J)
  {
    Eigen::NumericalDiff func(EigenFunctor<Func, Derived>(std::forward<Func>(f), ny_));
    func.df(x, J);
  }

private:
  int ny_;
};

}  // namespace ad_testing

#endif  // WRAPPERS__NUMERICAL_HPP_
