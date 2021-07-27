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


using namespace std::chrono_literals;

namespace ad_testing {

/**
 * @brief Compile-time for loop implementation
 */
template<typename _F, std::size_t... _Idx>
inline static constexpr auto static_for_impl(_F && f, std::index_sequence<_Idx...>)
{
  return (std::invoke(f, std::integral_constant<std::size_t, _Idx>()), ...);
}

/**
 * @brief Compile-time for loop over 0, ..., _I-1
 */
template<std::size_t _I, typename _F>
inline static constexpr auto static_for(_F && f)
{
  return static_for_impl(std::forward<_F>(f), std::make_index_sequence<_I>{});
}

// Check that two testers agree on a test
template<typename Tester1, typename Tester2, typename Test>
bool test_correctness(const Test & test)
{
  static constexpr Eigen::Index Nx = Test::InputSize;

  Eigen::Index nx = Nx == -1 ? test.input_size() : Nx;
  Eigen::Matrix<double, Nx, 1> x(nx);
  x.setOnes();
  Eigen::Index ny = test(x).size();

  Eigen::MatrixXd J1(ny, nx), J2(ny, nx);

  Tester1 tester1;
  Tester2 tester2;

  try {
    // setup
    tester1.setup(test, x);
    tester2.setup(test, x);

    // test
    for (auto i = 0u; i != 5u; i++) {
      x.setRandom();

      tester1.run(test, x, J1);
      tester2.run(test, x, J2);

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
SpeedResult test_speed(const Test & test)
{
  static constexpr Eigen::Index Nx = Test::InputSize;

  Eigen::Index nx = Nx == -1 ? test.input_size(): Nx;
  Eigen::Matrix<double, Nx, 1> x(nx);
  x.setOnes();
  Eigen::Index ny = test(x).size();

  Eigen::MatrixXd J1(ny, nx), J2(ny, nx);

  SpeedResult res{};

  Tester tester;

  std::atomic<bool> canceled = false;
  std::promise<std::tuple<std::string, std::size_t, std::chrono::nanoseconds>> setup_promise;
  auto setup_ftr = setup_promise.get_future();

  std::thread setup_thr([&]() {
    Eigen::Matrix<double, Nx, 1> x(nx);
    x.setOnes();
    try {
      std::size_t cntr = 0;
      const auto beg   = std::chrono::high_resolution_clock::now();
      while (!canceled) {
        tester.template setup(test, x);
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
  std::thread calc_thr([&]() {
    Eigen::Matrix<double, Nx, 1> x(nx);
    x.setOnes();
    Eigen::MatrixXd J(ny, nx);

    try {
      std::size_t cntr = 0;
      const auto beg   = std::chrono::high_resolution_clock::now();
      while (!canceled) {
        tester.template run(test, x, J);
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
