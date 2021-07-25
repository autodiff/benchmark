#ifndef WRAPPERS__AUTODIIF_VAR_HPP_
#define WRAPPERS__AUTODIIF_VAR_HPP_

#include <autodiff/common/meta.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <utility>

#include "common.hpp"

namespace ad_testing {

class AutodiffVarWrapper
{
public:
  static constexpr char name[] = "AutodiffVar";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(Func && f,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    auto x_ad = x.template cast<autodiff::var>().eval();
    auto F    = f(x_ad);

    if constexpr (EigenFunctor<Func, Derived>::JacobianType::RowsAtCompileTime != -1) {
      ::autodiff::detail::For<EigenFunctor<Func, Derived>::JacobianType::RowsAtCompileTime>(
        [&J, &F, &x_ad](auto i) { J.row(i) = autodiff::gradient(F(i), x_ad); });
    } else {
      for (auto i = 0u; i != F.rows(); ++i) { J.row(i) = autodiff::gradient(F(i), x_ad); }
    }
  }
};

}  // namespace ad_testing

#endif  // WRAPPERS__AUTODIIF_VAR_HPP_
