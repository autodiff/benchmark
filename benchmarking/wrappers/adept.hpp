#ifndef ADEPT_TESTER_HPP_
#define ADEPT_TESTER_HPP_

#include <adept.h>
#include <utility>

#include <Eigen/Core>

namespace Eigen {

template<>
struct NumTraits<adept::adouble> : NumTraits<double>
{
  typedef adept::adouble Real;
  typedef adept::adouble NonInteger;
  typedef adept::adouble Nested;

  enum {
    IsComplex             = 0,
    IsInteger             = 0,
    IsSigned              = 1,
    RequireInitialization = 1,
    ReadCost              = 1,
    AddCost               = 3,
    MulCost               = 3
  };
};

}  // namespace Eigen

namespace ad_testing {

class AdeptWrapper
{
public:
  static constexpr char name[] = "Adept";

  template<typename Func, typename Derived>
  void setup(Func && f, const Eigen::PlainObjectBase<Derived> & x)
  {}

  template<typename Func, typename Derived>
  void run(Func && f, const Eigen::PlainObjectBase<Derived> & x, Eigen::MatrixXd & J)
  {
    adept::Stack stack;

    auto ax = x.template cast<adept::adouble>().eval();

    stack.new_recording();

    auto ay = f(ax);

    stack.independent(ax.data(), ax.size());
    stack.dependent(ay.data(), ay.size());

    stack.jacobian(J.data());
  }
};

}  // namespace ad_testing

#endif  // ADEPT_TESTER_HPP_
