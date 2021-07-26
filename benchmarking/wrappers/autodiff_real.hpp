#ifndef WRAPPERS__AUTODIFF_REAL_HPP_
#define WRAPPERS__AUTODIFF_REAL_HPP_

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <utility>

namespace ad_testing {

class AutodiffRealWrapper
{
public:
  static constexpr char name[] = "AutodiffReal";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(Func && f, const Eigen::PlainObjectBase<Derived> & x, Eigen::MatrixXd & J)
  {
    auto x_ad = x.template cast<autodiff::real>().eval();
    J         = autodiff::jacobian(std::forward<Func>(f), autodiff::wrt(x_ad), autodiff::at(x_ad));
  }
};

}  // namespace ad_testing

#endif  // WRAPPERS__AUTODIFF_REAL_HPP_
