#ifndef WRAPPERS__SACADO_HPP_
#define WRAPPERS__SACADO_HPP_

#include <Eigen/Core>
#include <Sacado.hpp>
#include <utility>

namespace ad_testing {

class SacadoWrapper
{
public:
  static constexpr char name[] = "Sacado";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> &)
  {}

  template<typename Func, typename Derived>
  void run(Func && f, const Eigen::PlainObjectBase<Derived> & x, Eigen::MatrixXd & J)
  {
    Eigen::Matrix<Sacado::Fad::DFad<double>, Derived::RowsAtCompileTime, 1> x_ad(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      x_ad(i) = Sacado::Fad::DFad<double>(x.size(), i, x(i));
    }

    auto y_ad = f(x_ad);

    for (std::size_t j = 0; j < y_ad.size(); ++j) {
      for (std::size_t i = 0; i < x.size(); ++i) { J(j, i) = y_ad(j).dx(i); }
    }
  }
};

}  // namespace ad_testing

#endif  // WRAPPERS__SACADO_HPP_
