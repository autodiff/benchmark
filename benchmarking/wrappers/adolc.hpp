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
  void setup(Func && f, const Eigen::PlainObjectBase<Derived> & x)
  {
    if (adtl::getNumDir() != x.size()) { adtl::setNumDir(x.size()); }
    ny_ = f(x).size();
  }

  template<typename Func, typename Derived>
  void run(Func && f, const Eigen::PlainObjectBase<Derived> & x, Eigen::MatrixXd & J)
  {
    Eigen::AdolcForwardJacobian func(EigenFunctor<Func, Derived>(std::forward<Func>(f), ny_));
    typename EigenFunctor<Func, Derived>::ValueType y(ny_);
    func(x, &y, &J);
  }

private:
  int ny_;
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
    ny_ = f(x).size();

    EigenFunctor<Func, Derived> func(std::forward<Func>(f), ny_);

    trace_on(0);

    Eigen::Matrix<adouble, -1, 1> ax(x.size());
    Eigen::Matrix<adouble, -1, 1> ay(ny_);
    Eigen::Matrix<double, -1, 1> y(ny_);

    for (size_t i = 0; i < x.size(); i++) { ax(i) <<= x(i); }

    func(ax, &ay);

    for (size_t i = 0; i < ny_; i++) {
      y(i) = 0;
      ay(i) >>= y(i);
    }

    trace_off();
  }

  template<typename Func, typename Derived>
  void run(Func &&, const Eigen::PlainObjectBase<Derived> & x, Eigen::MatrixXd & J)
  {
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> Jrow(ny_, x.size());

    double ** jac_rows = new double *[ny_];
    for (size_t i = 0; i < ny_; i++) { jac_rows[i] = Jrow.row(i).data(); }

    // NOTE: if return value is negative we have to re-tape
    jacobian(0, ny_, x.size(), x.data(), jac_rows);

    delete[] jac_rows;
    J = Jrow;
  }

private:
  int ny_;
};

}  // namespace ad_testing

#endif  // WRAPPERS__ADOLC_HPP_
