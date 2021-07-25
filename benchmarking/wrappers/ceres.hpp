#ifndef WRAPPERS__CERES_HPP_
#define WRAPPERS__CERES_HPP_

#include <ceres/internal/autodiff.h>
#include <utility>

#include "common.hpp"

namespace ad_testing {

class CeresWrapper
{
public:
  static constexpr char name[] = "Ceres";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(Func && f,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    using Scalar = typename Derived::Scalar;

    EigenFunctor<Func, Derived> func(std::forward<Func>(f));
    typename EigenFunctor<Func, Derived>::JacobianTypeRowMajor Jrow;

    const Scalar * prms[1] = {x.data()};
    double ** jac_rows     = new double *[func.values()];
    for (size_t i = 0; i < func.values(); i++) { jac_rows[i] = Jrow.row(i).data(); }

    Scalar F{};
    ceres::internal::AutoDifferentiate<EigenFunctor<Func, Derived>::ValuesAtCompileTime,
      ceres::internal::StaticParameterDims<Derived::RowsAtCompileTime>>(
      func, prms, func.values(), &F, jac_rows);

    delete[] jac_rows;

    J = Jrow;
  }
};

}  // namespace ad_testing

#endif  // WRAPPERS__CERES_HPP_
