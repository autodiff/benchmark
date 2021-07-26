#ifndef TESTS_HPP_
#define TESTS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/numeric/odeint.hpp>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "se3.hpp"

using Eigen::Matrix, Eigen::MatrixBase;

namespace ad_testing {

/**
 * Return a constant vector of size N
 *
 * f: R -> R^N
 */
template<int _N>
struct Constant
{
  static constexpr char name[]   = "Constant";
  static constexpr int N         = _N;
  static constexpr int InputSize = 1;

  Constant(int n) : n_(n) {}

  int n() const { return n_; }
  int input_size() const { return 1; }

  template<typename D>
  Matrix<typename D::Scalar, N, 1> operator()(const MatrixBase<D> &) const
  {
    return Matrix<typename D::Scalar, N, 1>::Ones(n_);
  }

  int n_;
};

/**
 * Apply a series of coefficient-wise operations and sum the result
 *
 * f: R^N -> R
 */
template<int _N>
struct ManyToOne
{
  static constexpr char name[]   = "ManyToOne";
  static constexpr int N         = _N;
  static constexpr int InputSize = N;

  ManyToOne(int n) : n_(n) {}

  int n() const { return n_; }
  int input_size() const { return n_; }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 1, 1> operator()(
    const Eigen::MatrixBase<Derived> & x) const
  {
    return Eigen::Matrix<typename Derived::Scalar, 1, 1>(
      (x + x.cwiseInverse()).array().sin().matrix().sum());
  }

  int n_;
};

/**
 * Compute series x_{y+2} = sin(cos(x_y))
 *
 * f: R -> R^N
 */
template<int _N>
struct OneToMany
{
  static constexpr char name[]    = "OneToMany";
  static constexpr int N          = _N;
  static constexpr int InputSize  = 1;
  static constexpr int OutputSize = N == -1 ? -1 : N;

  OneToMany(int n) : n_(n) {}

  int n() const { return n_; }
  int input_size() const { return 1; }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, OutputSize, 1> operator()(
    const Eigen::MatrixBase<Derived> & x) const
  {
    using std::sin, std::cos;

    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, OutputSize, 1> ret(n_);
    ret(0) = x(0);
    for (int i = 0; i < n_ - 1; ++i) {
      if (i % 2 == 0) {
        ret(i + 1) = sin(ret(i));
      } else {
        ret(i + 1) = cos(ret(i));
      }
    }
    return ret;
  }

  int n_;
};

/**
 * Integrate an N-order integrator for 100 steps using a RK4 scheme
 *
 * f: R^N -> R^N
 */
template<int _N>
struct ODE
{
  static constexpr char name[]   = "ODE";
  static constexpr int N         = _N;
  static constexpr int InputSize = N;

  int n() const { return n_; }
  int input_size() const { return n_; }

  ODE(int n) : n_(n)
  {
    // matrix with ones on super-diagonal
    A_.setZero(n_, n_);
    A_.template block(0, 1, n_ - 1, n_ - 1).diagonal().setOnes();
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1> operator()(
    const Eigen::MatrixBase<Derived> & x) const
  {
    using scalar_t = typename Derived::Scalar;
    using state_t  = Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>;

    auto x0       = x.eval();
    const auto Ac = A_.template cast<scalar_t>().eval();

    boost::numeric::odeint::integrate_n_steps(
      boost::numeric::odeint::runge_kutta4<state_t,
        scalar_t,
        state_t,
        scalar_t,
        boost::numeric::odeint::vector_space_algebra>{},
      [&Ac](const state_t & x, state_t & dxdt, const scalar_t) { dxdt = Ac * x; },
      x0,
      scalar_t{0.},
      scalar_t{0.01},
      100);

    return x0;
  }

private:
  Eigen::Matrix<double, N, N> A_;
  int n_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Three-layer fully connected neural network with one channel and tanh activations
 *
 * Weights are statically allocated
 *
 * f: R^N -> R
 */
template<int _N>
struct NeuralNet
{
  static constexpr char name[]   = "NeuralNet";
  static constexpr int N         = _N;
  static constexpr int InputSize = N;

  int n() const { return n0_; }
  int input_size() const { return n0_; }

  NeuralNet(int n) : n0_(n)
  {
    n1_ = std::max<int>(1, n0_ / 2);
    n2_ = std::max<int>(1, n1_ / 2);
    n3_ = std::max<int>(1, n2_ / 2);

    std::srand(101);

    W1 = Eigen::Matrix<double, N1, N0>(n1_, n0_);
    W2 = Eigen::Matrix<double, N2, N1>(n2_, n1_);
    W3 = Eigen::Matrix<double, N3, N2>(n3_, n2_);

    W1.setRandom();
    W2.setRandom();
    W3.setRandom();
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 1, 1> operator()(
    const Eigen::MatrixBase<Derived> & x) const
  {
    const auto z1 = (W1.template cast<typename Derived::Scalar>() * x.normalized()).eval();
    const auto a1 = z1.array().tanh().matrix().eval();

    const auto z2 = (W2.template cast<typename Derived::Scalar>() * a1).eval();
    const auto a2 = z2.array().tanh().matrix().eval();

    const auto z3 = (W3.template cast<typename Derived::Scalar>() * a2).eval();
    const auto a3 = z3.array().tanh().matrix().eval();

    return Eigen::Matrix<typename Derived::Scalar, 1, 1>(
      (a3 - Eigen::Matrix<typename Derived::Scalar, N3, 1>::Ones(n3_)).squaredNorm());
  }

private:
  static constexpr int N0 = N;
  static constexpr int N1 = N == -1 ? -1 : std::max<int>(1, N0 / 2);
  static constexpr int N2 = N == -1 ? -1 : std::max<int>(1, N1 / 2);
  static constexpr int N3 = N == -1 ? -1 : std::max<int>(1, N2 / 2);

  int n0_, n1_, n2_, n3_;

  Eigen::Matrix<double, N1, N0> W1;
  Eigen::Matrix<double, N2, N1> W2;
  Eigen::Matrix<double, N3, N2> W3;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Camera reprojection error for N points
 *
 * f: R^6 -> R
 *
 * f(x) = \sum_i ( x_C_i - proj(CM * (P_CW * exp(x)) * x_W_i) ) .^ 2
 *
 * where - x_C_i the i:th 2d pixel point
 *       - x_W_i the i:th 3d world point
 *       - P_CW a nominal camera pose
 *       - x is a tangent space element defining an incremental pose
 */
template<int _N>
struct ReprojectionError
{
  static constexpr char name[]   = "Reprojection";
  static constexpr int N         = _N;
  static constexpr int InputSize = N == -1 ? -1 : 6;

  int n() const { return n_; }
  int input_size() const { return 6; }

  ReprojectionError(int n) : n_(n)
  {
    // nominal pose
    P_CW_nom = SE3<double>{Eigen::Quaterniond::Identity(), Eigen::Vector3d{0.1, -0.3, 0.2}};

    // camera matrix
    CM.setZero();
    CM(0, 0) = 700;  // fx
    CM(1, 1) = 690;  // fy
    CM(0, 2) = 320;  // cx
    CM(1, 2) = 240;  // cy
    CM(2, 2) = 1;

    // generate random data
    std::srand(101);

    pts_world.resize(n_);
    pts_image.resize(n_);

    for (int i = 0; i != n_; ++i) {
      pts_world[i]         = Eigen::Vector3d{0, 0, 3} + Eigen::Vector3d::Random();
      Eigen::Vector3d proj = CM * (P_CW_nom * pts_world[i]);
      pts_image[i]         = proj.template head<2>() / proj(2);
    }
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 1, 1> operator()(
    const Eigen::MatrixBase<Derived> & x) const
  {
    using Scalar = typename Derived::Scalar;
    using Vec3   = Eigen::Matrix<Scalar, 3, 1>;
    using Quat   = Eigen::Quaternion<Scalar>;

    SE3<Scalar> P_CW = SE3<Scalar>::exp(Scalar{0.01} * x) * P_CW_nom.template cast<Scalar>();
    auto CMc         = CM.template cast<Scalar>().eval();

    // Transform world points to camera frame, re-project, square
    Eigen::Matrix<Scalar, 1, 1> ret(0);
    for (int i = 0; i != n_; ++i) {
      Vec3 proj = CMc * (P_CW * pts_world[i].template cast<Scalar>().eval());
      ret(0) +=
        (proj.template head<2>() / proj(2) - pts_image[i].template cast<Scalar>()).squaredNorm();
    }
    return ret;
  }

private:
  int n_;
  Eigen::Matrix<double, 3, 3> CM;  // camera matrix
  SE3<double> P_CW_nom{};          // nominal camera pose
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pts_world;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts_image;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

/**
 * Differentiate the end effector position in an N link robotic arm
 *
 * f: R^6 -> R^6
 */
template<int _N>
struct Manipulator
{
  static constexpr char name[]   = "Manipulator";
  static constexpr int N         = _N;
  static constexpr int InputSize = N == -1 ? -1 : 6;

  int n() const { return n_; }
  int input_size() const { return 6; }

  Manipulator(int n) : n_(n)
  {
    // generate random link positions
    std::minstd_rand gen(101);  // fixed seed
    std::uniform_real_distribution<double> dis(-1, 1);
    auto gen_fcn = [&]() { return dis(gen); };

    link_pose.resize(n);

    for (int i = 0; i != n; ++i) {
      link_pose[i] = SE3<double>{Eigen::AngleAxis(M_PI_2 * dis(gen), Eigen::Vector3d::UnitX())
                                   * Eigen::AngleAxis(M_PI_2 * dis(gen), Eigen::Vector3d::UnitY())
                                   * Eigen::AngleAxis(M_PI_2 * dis(gen), Eigen::Vector3d::UnitZ()),
        Eigen::Vector3d{2 * dis(gen), 2 * dis(gen), 2 * dis(gen)}};
    }
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 3, 1> operator()(
    const Eigen::MatrixBase<Derived> & x) const
  {
    using Scalar  = typename Derived::Scalar;
    SE3<Scalar> P = SE3<Scalar>::exp(x);

    for (int i = 0; i != n_; ++i) { P *= link_pose[i].template cast<Scalar>(); }

    return P * Eigen::Matrix<Scalar, 3, 1>::UnitX().eval();
  }

private:
  int n_;
  std::vector<SE3<double>, Eigen::aligned_allocator<SE3<double>>> link_pose;
};

/**
 * Integrate a system on SE(3) for N steps using the RK4 scheme
 *
 * f: R^6 -> R^6
 */
template<int _N>
struct SE3ODE
{
  static constexpr char name[]   = "SE3ODE";
  static constexpr int N         = _N;
  static constexpr int InputSize = N == -1 ? -1 : 6;

  int n() const { return n_; }
  int input_size() const { return 6; }

  SE3ODE(int n) : n_(n)
  {
    velocity << 0.1, -0.2, 0.3, 0.1, -0.2, 0.3;
    Pfinal = SE3<double>{} * SE3<double>::exp(static_cast<double>(n_) * 0.01 * velocity);
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 6, 1> operator()(
    const Eigen::MatrixBase<Derived> & x) const
  {
    using scalar_t = typename Derived::Scalar;
    using state_t  = SE3<scalar_t>;
    using deriv_t  = typename state_t::Tangent;

    const auto vel_c = velocity.template cast<scalar_t>().eval();

    // set initial pose
    SE3<scalar_t> P = SE3<scalar_t>::exp(x);

    boost::numeric::odeint::integrate_n_steps(
      boost::numeric::odeint::runge_kutta4<state_t,
        scalar_t,
        deriv_t,
        scalar_t,
        boost::numeric::odeint::vector_space_algebra,
        lie_operations>{},
      [&vel_c](const state_t & X, deriv_t & dXdt, const scalar_t) { dXdt = vel_c; },
      P,
      scalar_t{0.},
      scalar_t{0.01},
      n_);

    const SE3<scalar_t> Pfinalinv = Pfinal.inv().template cast<scalar_t>();
    return (Pfinalinv * P).log();
  }

private:
  int n_;
  Eigen::Matrix<double, 6, 1> velocity;
  SE3<double> Pfinal{};

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace ad_testing

#endif  // TESTS_HPP_
