#ifndef WRAPPERS__CERES_HPP_
#define WRAPPERS__CERES_HPP_

#include <ceres/internal/autodiff.h>
#include <utility>

namespace ad_testing {

class CeresWrapper
{
public:
  static constexpr char name[] = "Ceres";

  template<typename Func, typename Derived>
  void setup(Func && f, const Eigen::PlainObjectBase<Derived> & x)
  {
    ny_ = f(x).size();
  }

  template<typename Func, typename Derived>
  void run(Func && f, const Eigen::PlainObjectBase<Derived> & x, Eigen::MatrixXd & J)
  {
    static constexpr Eigen::Index Nx = Derived::SizeAtCompileTime;
    static constexpr Eigen::Index Ny =
      std::invoke_result_t<Func, Eigen::Matrix<double, Nx, 1>>::SizeAtCompileTime;

    Eigen::Index nx = x.size();

    Eigen::Matrix<double, Ny, Nx, Nx == 1 ? Eigen::ColMajor : Eigen::RowMajor> Jrow(ny_, nx);

    Eigen::Matrix<double, Ny, 1> y(ny_);
    const double * parameter_ptrs[1] = {x.data()};
    double * jacobian_ptrs[1]        = {Jrow.data()};

    if constexpr (Nx == -1) {
      // not handled for now

      // Dynamic sizes not suppored in AutoDiffereniate, see
      // https://github.com/ceres-solver/ceres-solver/blob/ec4f2995bbde911d6861fb5c9bb7353ad796e02b/include/ceres/dynamic_autodiff_cost_function.h#L112

      /* ceres::internal::AutoDifferentiate<Ny, ceres::internal::DynamicParameterDims>(
        [&f, &nx, ny = ny_](const auto * const * in, auto * out) {
          std::cout << "Entering" << std::endl;
          using T = std::remove_pointer_t<std::decay_t<decltype(out)>>;
          Eigen::Map<const Eigen::Matrix<T, Nx, 1>> x_map(in[0], nx);
          std::cout << x_map << std::endl;
          Eigen::Map<Eigen::Matrix<T, Ny, 1>> y_map(out, ny);
          y_map = f(x_map);
          std::cout << y_map << std::endl;
          return true;
        },
        parameter_ptrs,
        ny_,
        y.data(),
        jacobian_ptrs); */
    } else {
      ceres::internal::AutoDifferentiate<Ny, ceres::internal::StaticParameterDims<Nx>>(
        [&f, &nx, ny = ny_](const auto * in, auto * out) {
          using T = std::remove_pointer_t<std::decay_t<decltype(out)>>;
          Eigen::Map<const Eigen::Matrix<T, Nx, 1>> x_map(in, nx);
          Eigen::Map<Eigen::Matrix<T, Ny, 1>> y_map(out, ny);
          y_map = f(x_map);
          return true;
        },
        parameter_ptrs,
        ny_,
        y.data(),
        jacobian_ptrs);
    }

    J = Jrow;
  }

private:
  int ny_;
};

}  // namespace ad_testing

#endif  // WRAPPERS__CERES_HPP_
