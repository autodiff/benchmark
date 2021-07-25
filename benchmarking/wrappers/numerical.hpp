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
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(Func && f,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    Eigen::NumericalDiff func(EigenFunctor<Func, Derived>(std::forward<Func>(f)));
    func.df(x, J);
  }
};

}  // namespace ad_testing

#endif  // WRAPPERS__NUMERICAL_HPP_
