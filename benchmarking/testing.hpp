#ifndef TESTING_HPP_
#define TESTING_HPP_

#include <Eigen/Core>

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#include "wrappers/common.hpp"


using namespace std::chrono_literals;

namespace ad_testing {

// Static for loop
template<size_t i, size_t ibegin, size_t iend, typename Function>
constexpr auto AuxStaticFor(Function && f)
{
  if constexpr (i < iend) {
    f(std::integral_constant<size_t, i>{});
    AuxStaticFor<i + 1, ibegin, iend>(std::forward<Function>(f));
  }
}

template<size_t ibegin, size_t iend, typename Function>
constexpr auto StaticFor(Function && f)
{
  AuxStaticFor<ibegin, ibegin, iend>(std::forward<Function>(f));
}

template<size_t iend, typename Function>
constexpr auto StaticFor(Function && f)
{
  StaticFor<0, iend>(std::forward<Function>(f));
}

// Helper type to holds list of types
template<typename... T>
struct TypePack
{
  static constexpr std::size_t size = sizeof...(T);

  template<std::size_t Idx>
  using type = std::tuple_element_t<Idx, std::tuple<T...>>;
};

// Check that two testers agree on a test
template<typename Tester1, typename Tester2, typename Test>
bool test_correctness()
{
  Tester1 tester1;
  Tester2 tester2;
  Test test;

  Eigen::Matrix<double, Test::InputSize, 1> x = Eigen::Matrix<double, Test::InputSize, 1>::Ones();
  typename EigenFunctor<Test, decltype(x)>::JacobianType J1, J2;
  try {
    // setup
    tester1.setup([&test](const auto & x) { return test(x); }, x);
    tester2.setup([&test](const auto & x) { return test(x); }, x);

    // test
    for (size_t i = 0; i < 5; i++) {
      x = Eigen::Matrix<double, Test::InputSize, 1>::Random();

      tester1.run([&test](const auto & x) { return test(x); }, x, J1);
      tester2.run([&test](const auto & x) { return test(x); }, x, J2);

      if (!J1.isApprox(J2, 1e-2)) {
        std::cerr << "Different jacobians detected on " << Test::name << std::endl;
        std::cerr << "Jacobian from " << Tester1::name << std::endl;
        std::cerr << J1 << std::endl;
        std::cerr << "Jacobian from " << Tester2::name << std::endl;
        std::cerr << J2 << std::endl;
        return false;
      }
    }
  } catch (const std::exception & e) {
    std::cerr << "Exception thrown during correctness test: " << e.what() << '\n';
    return false;
  }

  return true;
}

// Result for speed test
struct SpeedResult
{
  std::string exception{""};
  bool setup_timeout{false}, calc_timeout{false};
  uint64_t setup_iter{}, calc_iter{};
  std::chrono::nanoseconds setup_time{}, calc_time{};
};

// Speed-test a Tester on a Test
template<typename Tester, typename Test>
SpeedResult test_speed()
{
  SpeedResult res{};

  Tester tester;
  Test test;

  std::atomic<bool> canceled = false;
  std::promise<std::tuple<std::string, std::size_t, std::chrono::nanoseconds>> setup_promise;
  auto setup_ftr = setup_promise.get_future();

  std::thread setup_thr([&tester, &test, &canceled, &setup_promise]() {
    Eigen::Matrix<double, Test::InputSize, 1> x;
    x.setOnes();
    try {
      std::size_t cntr = 0;
      const auto beg   = std::chrono::high_resolution_clock::now();
      while (!canceled) {
        tester.template setup([&test](const auto & x) { return test(x); }, x);
        ++cntr;
      }
      const auto end = std::chrono::high_resolution_clock::now();

      setup_promise.set_value(std::make_tuple(std::string{}, cntr, end - beg));
    } catch (const std::exception & e) {
      setup_promise.set_value(std::make_tuple(e.what(), 0, 0ns));
    }
  });

  std::this_thread::sleep_for(500ms);
  canceled.store(true);

  if (setup_ftr.wait_for(20s) == std::future_status::ready) {
    setup_thr.join();
    std::tie(res.exception, res.setup_iter, res.setup_time) = setup_ftr.get();
  } else {
    setup_thr.detach();
    res.setup_timeout = true;
    return res;
  }

  if (!res.exception.empty()) { return res; }

  std::promise<std::tuple<std::string, std::size_t, std::chrono::nanoseconds>> calc_promise;
  auto calc_ftr = calc_promise.get_future();
  canceled.store(false);
  std::thread calc_thr([&test, &tester, &canceled, &calc_promise]() {
    Eigen::Matrix<double, Test::InputSize, 1> x;
    x.setOnes();
    typename EigenFunctor<Test, decltype(x)>::JacobianType J;

    try {
      std::size_t cntr = 0;
      const auto beg   = std::chrono::high_resolution_clock::now();
      while (!canceled) {
        tester.template run([&test](const auto & x) { return test(x); }, x, J);
        ++cntr;
      }
      const auto end = std::chrono::high_resolution_clock::now();
      calc_promise.set_value(std::make_tuple(std::string{}, cntr, end - beg));
    } catch (const std::exception & e) {
      calc_promise.set_value(std::make_tuple(e.what(), 0, 0ns));
    }
  });

  std::this_thread::sleep_for(3s);
  canceled.store(true);

  if (calc_ftr.wait_for(20s) == std::future_status::ready) {
    calc_thr.join();
    std::tie(res.exception, res.calc_iter, res.calc_time) = calc_ftr.get();
  } else {
    calc_thr.detach();  // forceful termination
    res.calc_timeout = true;
  }

  return res;
}

}  // namespace ad_testing

#endif  // TESTING_HPP_
