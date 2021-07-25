#ifndef WRAPPERS__AUTODIFF_DUAL_HPP_
#define WRAPPERS__AUTODIFF_DUAL_HPP_

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <utility>

#include "common.hpp"

namespace ad_testing {

class AutodiffDualWrapper
{
public:
  static constexpr char name[] = "AutodiffDual";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(Func && f,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    auto x_ad = x.template cast<autodiff::dual>().eval();
    J         = autodiff::jacobian(std::forward<Func>(f), autodiff::wrt(x_ad), autodiff::at(x_ad));
  }
};

}  // namespace ad_testing

#endif  // WRAPPERS__AUTODIFF_DUAL_HPP_
