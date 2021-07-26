#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <Eigen/Core>

#include <utility>

namespace ad_testing {

/**
 * @brief Functor that wraps a Test in a functor with multiple forms of operator()
 */
template<typename _Test, typename _InputType>
struct EigenFunctor
{
  using Scalar    = typename _InputType::Scalar;
  using InputType = _InputType;
  using ValueType = std::invoke_result_t<_Test, InputType>;

  static constexpr int InputsAtCompileTime = InputType::SizeAtCompileTime;
  static constexpr int ValuesAtCompileTime = ValueType::SizeAtCompileTime;

  using JacobianType = Eigen::MatrixXd;

  explicit EigenFunctor(_Test && test, int ny) : test_(std::forward<_Test>(test)), ny_(ny) {}

  /**
   * @brief Dimension of output vector
   */
  int values() const { return ny_; }

  /**
   * @brief Functor form used in unsupported/AdolcForward
   */
  template<typename Derived1, typename Derived2>
  void operator()(const Eigen::MatrixBase<Derived1> & x, Eigen::MatrixBase<Derived2> * y) const
  {
    *y = test_(x);
  }

  /**
   * @brief Functor form used in unsupported/NumericalDiff
   */
  template<typename Derived1, typename Derived2>
  void operator()(const Eigen::MatrixBase<Derived1> & x, Eigen::MatrixBase<Derived2> & y) const
  {
    y = test_(x);
  }

private:
  _Test test_;
  int ny_;
};

}  // namespace ad_testing

#endif  // COMMON_HPP_
