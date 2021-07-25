#ifndef SE3_HPP_
#define SE3_HPP_

#include <Eigen/Geometry>
#include <iostream>

namespace ad_testing {

/**
 * @brief Bare-bones implementation of the SE(3) Lie Group
 *
 * For benchmarking purposes only---do not use in important code.
 */
template<typename _Scalar>
struct SE3
{
  using Scalar  = _Scalar;
  using Tangent = Eigen::Matrix<Scalar, 6, 1>;

  // group composition
  SE3<Scalar> operator*(const SE3<Scalar> & other) const
  {
    return SE3<Scalar>{q * other.q, t + q * other.t};
  }

  // in-place group composition
  SE3<Scalar> & operator*=(const SE3<Scalar> & other)
  {
    t += q * other.t;
    q *= other.q;
    return *this;
  }

  // group action on 3-vector
  template<typename Derived>
  Derived operator*(const Eigen::EigenBase<Derived> & vec) const
  {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3)

    return t + q * vec;
  }

  // group inverse
  SE3<Scalar> inv() const { return SE3<Scalar>{q.inverse(), -(q.inverse() * t)}; }

  // cast to different scalar type
  template<typename NewScalar>
  SE3<NewScalar> cast() const
  {
    return SE3<NewScalar>{q.template cast<NewScalar>(), t.template cast<NewScalar>()};
  }

  // exponential (does not work for special case of zero orientation (yields 0/0))
  template<typename Derived>
  static SE3<Scalar> exp(const Eigen::MatrixBase<Derived> & tangent)
  {
    EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Tangent, Derived)

    using std::sin, std::cos;

    const Scalar w_norm                        = tangent.template tail<3>().norm();
    const Eigen::Matrix<Scalar, 3, 1> q_diff_n = tangent.template tail<3>().normalized();

    Eigen::Matrix<Scalar, 3, 3> what;
    what << Scalar(0), -q_diff_n(2), q_diff_n(1), q_diff_n(2), Scalar(0), -q_diff_n(0),
      -q_diff_n(1), q_diff_n(0), Scalar(0);

    const Eigen::Matrix<Scalar, 3, 3> J = Eigen::Matrix<Scalar, 3, 3>::Identity()
                                        + ((w_norm - sin(w_norm)) / w_norm) * what * what
                                        + ((Scalar(1.) - cos(w_norm)) / w_norm) * what;

    return SE3<Scalar>{Eigen::Quaternion<Scalar>(Eigen::AngleAxis<Scalar>(w_norm, q_diff_n)),
      J * tangent.template head<3>()};
  }

  // log (does not work for special case of zero orientation (yields 0/0))
  Tangent log() const
  {
    using std::atan2;

    const Scalar alpha = atan2(q.coeffs().template head<3>().norm(), q.w());

    const Eigen::Matrix<Scalar, 3, 1> so3log =
      Scalar(2.) * (alpha / sin(alpha)) * q.coeffs().template head<3>();

    // TODO: could derive more efficient expression for the inverse...
    const Scalar w_norm                        = so3log.norm();
    const Eigen::Matrix<Scalar, 3, 1> q_diff_n = so3log.normalized();

    Eigen::Matrix<Scalar, 3, 3> what;
    what << Scalar(0), -q_diff_n(2), q_diff_n(1), q_diff_n(2), Scalar(0), -q_diff_n(0),
      -q_diff_n(1), q_diff_n(0), Scalar(0);

    const Eigen::Matrix<Scalar, 3, 3> J = Eigen::Matrix<Scalar, 3, 3>::Identity()
                                        + ((w_norm - sin(w_norm)) / w_norm) * what * what
                                        + ((Scalar(1.) - cos(w_norm)) / w_norm) * what;

    return (Tangent() << J.inverse() * t, so3log).finished();
  }

  Eigen::Quaternion<Scalar> q   = Eigen::Quaternion<Scalar>::Identity();
  Eigen::Matrix<Scalar, 3, 1> t = Eigen::Matrix<Scalar, 3, 1>::Zero();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief boost::numerical::odeint operations for numerical integration on lie groups
 */
struct lie_operations
{
  template<class Fac1 = double, class Fac2 = Fac1>
  struct scale_sum2
  {
    const Fac2 m_alpha2;

    scale_sum2(Fac1 alpha1, Fac2 alpha2) : m_alpha2(alpha2)
    {
      if (alpha1 != Fac1(1.0)) {
        std::cerr << "alpha1 != 1 not expected for Lie type" << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    template<class T1, class T2, class T3>
    void operator()(T1 & t1, const T2 & t2, const T3 & t3) const
    {
      t1 = t2 * T2::exp(t3 * m_alpha2);
    }

    typedef void result_type;
  };

  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2>
  struct scale_sum3
  {
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;

    scale_sum3(Fac1 alpha1, Fac2 alpha2, Fac3 alpha3) : m_alpha2(alpha2), m_alpha3(alpha3)
    {
      if (alpha1 != Fac1(1.0)) {
        std::cerr << "alpha1 != 1 not expected for Lie type" << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    template<class T1, class T2, class T3, class T4>
    void operator()(T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4) const
    {
      t1 = t2 * T2::exp(t3 * m_alpha2 + t4 * m_alpha3);
    }

    typedef void result_type;
  };

  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3>
  struct scale_sum4
  {
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;

    scale_sum4(Fac1 alpha1, Fac2 alpha2, Fac3 alpha3, Fac4 alpha4)
        : m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4)
    {
      if (alpha1 != Fac1(1.0)) {
        std::cerr << "alpha1 != 1 not expected for Lie type" << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    template<class T1, class T2, class T3, class T4, class T5>
    void operator()(T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5) const
    {
      t1 = t2 * T2::exp(t3 * m_alpha2 + t4 * m_alpha3 + t5 * m_alpha4);
    }

    typedef void result_type;
  };

  template<class Fac1 = double,
    class Fac2        = Fac1,
    class Fac3        = Fac2,
    class Fac4        = Fac3,
    class Fac5        = Fac4>
  struct scale_sum5
  {
    const Fac2 m_alpha2;
    const Fac3 m_alpha3;
    const Fac4 m_alpha4;
    const Fac5 m_alpha5;

    scale_sum5(Fac1 alpha1, Fac2 alpha2, Fac3 alpha3, Fac4 alpha4, Fac5 alpha5)
        : m_alpha2(alpha2), m_alpha3(alpha3), m_alpha4(alpha4), m_alpha5(alpha5)
    {
      if (alpha1 != Fac1(1.0)) {
        std::cerr << "alpha1 != 1 not expected for Lie type" << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    template<class T1, class T2, class T3, class T4, class T5, class T6>
    void operator()(
      T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5, const T6 & t6) const
    {
      t1 = t2 * T2::exp(t3 * m_alpha2 + t4 * m_alpha3 + t5 * m_alpha4 + t6 * m_alpha5);
    }

    typedef void result_type;
  };
};

}  // namespace ad_testing

#endif  // SE3_HPP_
