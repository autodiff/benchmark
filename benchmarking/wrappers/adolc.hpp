#ifndef WRAPPERS__ADOLC_HPP_
#define WRAPPERS__ADOLC_HPP_

#include <utility>

#include <adolc/adolc.h>
#include <unsupported/Eigen/AdolcForward>

#include "common.hpp"

namespace ad_testing {

/**
 * @brief Use ADOL-C in no-taping forward mode
 */
class AdolcTapelessWrapper
{
public:
  static constexpr char name[] = "ADOL-C-Tapeless";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> & x)
  {
    if (adtl::getNumDir() != x.size()) { adtl::setNumDir(x.size()); }
  }

  template<typename Func, typename Derived>
  void run(Func && f,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    Eigen::AdolcForwardJacobian func(EigenFunctor<Func, Derived>(std::forward<Func>(f)));
    typename EigenFunctor<Func, Derived>::ValueType y;
    func(x, &y, &J);
  }
};

/**
 * @brief Use ADOL-C in classical taping mode
 *
 * Segfaults on OneToMany (reason not investigated)
 */
class AdolcWrapper
{
public:
  static constexpr char name[] = "ADOL-C";

  template<typename Func, typename Derived>
  void setup(Func && f, const Eigen::PlainObjectBase<Derived> & x)
  {
    EigenFunctor<Func, Derived> func(std::forward<Func>(f));

    num_outputs = func.values();

    trace_on(0);

    adouble * ax = new adouble[x.size()];
    double * y   = new double[num_outputs];
    adouble * ay = new adouble[num_outputs];

    for (size_t i = 0; i < x.size(); i++) { ax[i] <<= x(i); }

    func(ax, ay);

    for (size_t i = 0; i < num_outputs; i++) {
      y[i] = 0;
      ay[i] >>= y[i];
    }

    trace_off();

    delete[] ax;
    delete[] y;
    delete[] ay;
  }

  template<typename Func, typename Derived>
  void run(Func &&,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    typename EigenFunctor<Func, Derived>::JacobianTypeRowMajor Jrow(num_outputs, x.size());

    double ** jac_rows = new double *[num_outputs];
    for (size_t i = 0; i < num_outputs; i++) { jac_rows[i] = Jrow.row(i).data(); }

    // NOTE: if return value is negative we have to re-tape
    jacobian(0, num_outputs, x.size(), x.data(), jac_rows);

    delete[] jac_rows;
    J = Jrow;
  }

private:
  int num_outputs;
};

}  // namespace ad_testing

#endif  // WRAPPERS__ADOLC_HPP_
