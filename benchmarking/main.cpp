#include <Eigen/Core>

#include <iomanip>
#include <iostream>
#include <tuple>

#include "wrappers/numerical.hpp"
#ifdef ENABLE_ADEPT
#include "wrappers/adept.hpp"
#endif
#ifdef ENABLE_ADOLC
#include "wrappers/adolc.hpp"
#endif
#ifdef ENABLE_AUTODIFF_DUAL
#include "wrappers/autodiff_dual.hpp"
#endif
#ifdef ENABLE_AUTODIFF_REAL
#include "wrappers/autodiff_real.hpp"
#endif
#ifdef ENABLE_AUTODIFF_VAR
#include "wrappers/autodiff_var.hpp"
#endif
#ifdef ENABLE_CERES
#include "wrappers/ceres.hpp"
#endif
#ifdef ENABLE_CERES
#include "wrappers/ceres.hpp"
#endif
#ifdef ENABLE_CPPADCG
#include "wrappers/cppadcg.hpp"  // must be before cppad
#endif
#ifdef ENABLE_CPPAD
#include "wrappers/cppad.hpp"
#endif
#ifdef ENABLE_SACADO
#include "wrappers/sacado.hpp"
#endif

#include "testing.hpp"
#include "tests.hpp"

using namespace ad_testing;

template<typename Tester, typename Test>
void run_speedtest(const Test & test)
{
  auto res = ad_testing::test_speed<Tester>(test);

  std::string name = std::string(Test::name)
    + "<" + std::to_string(Test::N) + ">"
    + "(" + std::to_string(test.n()) + ")";

  if (res.calc_timeout || res.setup_timeout) {
    std::cerr << std::left << std::setw(20) << Tester::name << std::left << std::setw(20) << name
              << "TIMEOUT (DETACHED)" << std::endl;
    return;
  }

  // check if error occured
  if (!res.exception.empty()) {
    std::cerr << std::left << std::setw(20) << Tester::name << std::left << std::setw(20) << name
              << "EXCEPTION: " << res.exception << std::endl;
    return;
  }

  // compare with numerical
  if (!ad_testing::test_correctness<Tester, NumericalWrapper>(test)) {
    std::cerr << std::left << std::setw(20) << Tester::name << std::left << std::setw(20) << name
              << "CORRECTNESS ERROR" << std::endl;
  }

  std::cout << std::left << std::setw(20) << Tester::name << std::left << std::setw(20) << name
            << std::setprecision(2) << std::right << std::setw(10) << std::scientific
            << static_cast<double>(res.setup_time.count()) / res.setup_iter << std::right
            << std::setw(10) << std::scientific
            << static_cast<double>(res.calc_time.count()) / res.calc_iter << std::endl;
}

template<typename Tester, typename ... Test>
void run_tests(const std::tuple<Test...> & tests)
{
  ad_testing::static_for<std::tuple_size_v<std::decay_t<decltype(tests)>>>([&](auto i) {
    run_speedtest<Tester>(std::get<i>(tests));
  });
}

int main()
{
  auto all_tests = std::make_tuple(
    Constant<3>(3),
    Constant<-1>(3),
    ManyToOne<3>(3),
    ManyToOne<-1>(3)
    /* Constant<10>,
    Constant<100>,
    Constant<1000>,
    ManyToOne<3>,
    ManyToOne<10>,
    ManyToOne<100>,
    ManyToOne<1000>,
    OneToMany<3>,
    OneToMany<10>,
    OneToMany<100>,
    OneToMany<1000>,
    ODE<3>,
    ODE<10>,
    ODE<30>,
    NeuralNet<3>,
    NeuralNet<10>,
    NeuralNet<100>,
    ReprojectionError<3>,
    ReprojectionError<10>,
    ReprojectionError<100>,
    ReprojectionError<1000>,
    Manipulator<3>,
    Manipulator<10>,
    Manipulator<100>,
    Manipulator<1000>,
    SE3ODE<3>,
    SE3ODE<10>,
    SE3ODE<100>,
    SE3ODE<1000> */
  );

  auto tests_no_ode = std::make_tuple(
    Constant<3>(3),
    /* Constant<10>(10),
    Constant<100>(100),
    Constant<1000>(1000), */
    ManyToOne<3>(3)
    /* ManyToOne<100>,
    ManyToOne<1000>,
    OneToMany<3>,
    OneToMany<10>,
    OneToMany<100>,
    OneToMany<1000>,
    NeuralNet<3>,
    NeuralNet<10>,
    NeuralNet<100>,
    ReprojectionError<3>,
    ReprojectionError<10>,
    ReprojectionError<100>,
    ReprojectionError<1000>,
    Manipulator<3>,
    Manipulator<10>,
    Manipulator<100>,
    Manipulator<1000> */
  );

  auto tests_no_se3 = std::make_tuple(
    Constant<3>(3),
    Constant<-1>(3)
    /* Constant<10>,
    Constant<100>,
    Constant<1000>,
    ManyToOne<3>,
    ManyToOne<10>,
    ManyToOne<100>,
    ManyToOne<1000>,
    OneToMany<3>,
    OneToMany<10>,
    OneToMany<100>,
    OneToMany<1000>,
    ODE<3>,
    ODE<10>,
    ODE<30>,
    NeuralNet<3>,
    NeuralNet<10>,
    NeuralNet<100> */
  );

  // these can run all tests (but do not necessarily succeed)
#ifdef ENABLE_ADEPT
  run_tests<AdeptWrapper>(all_tests);
#endif
#ifdef ENABLE_ADOLC
  run_tests<AdolcTapelessWrapper>(all_tests);
  // Adolc taped can not handle Eigen quaterion-vector product
  run_tests<AdolcWrapper>(tests_no_se3);
#endif
#ifdef ENABLE_AUTODIFF_DUAL
  run_tests<AutodiffDualWrapper>(all_tests);
#endif
#ifdef ENABLE_AUTODIFF_REAL
  // Real can not handle atan2
  run_tests<AutodiffRealWrapper>(tests_no_se3);
#endif
#ifdef ENABLE_AUTODIFF_VAR
  // * AutodiffRev times out on ODE tests
  run_tests<AutodiffVarWrapper>(tests_no_ode);
#endif
#ifdef ENABLE_CPPAD
  run_tests<CppADWrapper>(all_tests);
#endif
#ifdef ENABLE_CPPADCG
  run_tests<CppADCGWrapper>(all_tests);
#endif
#ifdef ENABLE_NUMERICAL
  run_tests<NumericalWrapper>(all_tests);
#endif
#ifdef ENABLE_SACADO
  run_tests<SacadoWrapper>(all_tests);
#endif
#ifdef ENABLE_CERES
  // * Ceres can't handle ODE tests because no conversion double -> Jet
  run_tests<CeresWrapper>(tests_no_ode);
#endif

  return 0;
}
