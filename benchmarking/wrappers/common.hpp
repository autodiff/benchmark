#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <Eigen/Core>

#include <utility>

namespace ad_testing {

/**
 * @brief Functor that wraps a callable with multiple forms of operator()
 */
template<typename _Func, typename _InputType>
struct EigenFunctor
{
  using Scalar    = typename _InputType::Scalar;
  using InputType = _InputType;
  using ValueType = std::invoke_result_t<_Func, InputType>;

  static constexpr int InputsAtCompileTime = InputType::SizeAtCompileTime;
  static constexpr int ValuesAtCompileTime = ValueType::SizeAtCompileTime;

  using JacobianType =
    Eigen::Matrix<Scalar, ValueType::SizeAtCompileTime, InputType::SizeAtCompileTime>;

  using JacobianTypeRowMajor = Eigen::Matrix<Scalar,
    ValueType::SizeAtCompileTime,
    InputType::SizeAtCompileTime,
    InputType::SizeAtCompileTime == 1 ? Eigen::ColMajor : Eigen::RowMajor>;

  int values_ = ValueType::SizeAtCompileTime;  // must be changed for dynamic sizing

  explicit EigenFunctor(_Func && func) : func_(std::forward<_Func>(func)) {}

  /**
   * @brief Return (dynamic) dimension of output vector
   */
  int values() const { return values_; }

  /**
   * @brief Functor form used in unsupported/AdolcForward
   */
  template<typename Derived1, typename Derived2>
  void operator()(const Eigen::MatrixBase<Derived1> & x, Eigen::MatrixBase<Derived2> * y) const
  {
    *y = func_(x);
  }

  /**
   * @brief Functor form used in unsupported/NumericalDiff
   */
  template<typename Derived1, typename Derived2>
  void operator()(const Eigen::MatrixBase<Derived1> & x, Eigen::MatrixBase<Derived2> & y) const
  {
    y = func_(x);
  }

  /**
   * @brief Functor form used in Ceres
   */
  template<typename T>
  bool operator()(const T * x, T * y) const
  {
    // TODO: expose dynamic input size and add here
    Eigen::Map<const Eigen::Matrix<T, InputsAtCompileTime, 1>> x_map(x);
    Eigen::Map<Eigen::Matrix<T, ValuesAtCompileTime, 1>> y_map(y, values());

    y_map = func_(x_map);

    return true;
  }

private:
  _Func func_;
};

}  // namespace ad_testing

#endif  // COMMON_HPP_
