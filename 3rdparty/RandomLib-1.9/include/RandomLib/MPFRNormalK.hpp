/**
 * \file MPFRNormalK.hpp
 * \brief Header for MPFRNormalK
 *
 * Sampling exactly from the normal distribution for MPFR.
 *
 * Copyright (c) Charles Karney (2012) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_MPFRNORMALK_HPP)
#define RANDOMLIB_MPFRNORMALK_HPP 1

#include <algorithm>            // for max
#include <RandomLib/MPFRRandom.hpp>
#include <RandomLib/MPFRExponential.hpp>

#if HAVE_MPFR || defined(DOXYGEN)

namespace RandomLib {

  /**
   * \brief The normal distribution for MPFR (Kahn algorithm).
   *
   * This class is <b>DEPRECATED</b>.  It is included for illustrative purposes
   * only.  The MPFRNormal class provides a somewhat more efficient method for
   * sampling from the normal distribution.
   *
   * Refs:
   * - H. Kahn, Rand Report RM-1237-AEC, p. 41 (1954).
   * - M. Abramowitz and I. A. Stegun, p. 953, Sec. 26.8.6.a(4) (1964).
   * .
   * N.B. Damien Stehle' drew my attention to this algorithm as a useful way to
   * compute normal deviates exactly.
   *
   * This class uses mutable private objects.  So a single MPFRNormalK object
   * cannot safely be used by multiple threads.  In a multi-processing
   * environment, each thread should use a thread-specific MPFRNormalK object.
   *
   * @tparam bits the number of bits in each digit.
   **********************************************************************/
  template<int bits = 32> class MPFRNormalK {
  public:

    /**
     * Initialize the MPFRNormalK object.
     **********************************************************************/
    MPFRNormalK()
    { mpfr_init2(_xf, MPFR_PREC_MIN); mpfr_init2(_zf, MPFR_PREC_MIN); }
    /**
     * Destroy the MPFRNormalK object.
     **********************************************************************/
    ~MPFRNormalK()
    { mpfr_clear(_zf); mpfr_clear(_xf); }
    /**
     * Sample from the normal distribution with mean 0 and variance 1 returning
     * a MPFRRandom.
     *
     * @param[out] t the MPFRRandom result.
     * @param[in,out] r a GMP random generator.
     **********************************************************************/
    void operator()(MPFRRandom<bits>& t, gmp_randstate_t r) const
    { Compute(r); _x.swap(t); }
    /**
     * Sample from the normal distribution with mean 0 and variance 1.
     *
     * @param[out] val the sample from the normal distribution
     * @param[in,out] r a GMP random generator.
     * @param[in] round the rounding direction.
     * @return the MPFR ternary result (&plusmn;1 if val is larger/smaller than
     *   the exact sample).
     **********************************************************************/
    int operator()(mpfr_t val, gmp_randstate_t r, mpfr_rnd_t round) const
    { Compute(r); return _x(val, r, round); }
  private:
    // disable copy constructor and assignment operator
    MPFRNormalK(const MPFRNormalK&);
    MPFRNormalK& operator=(const MPFRNormalK&);
    void Compute(gmp_randstate_t r) const {
      // The algorithm is sample x and z from the exponential distribution; if
      // (x-1)^2 < 2*z, return (random sign)*x; otherwise repeat.  Probability
      // of acceptance is sqrt(pi/2) * exp(-1/2) = 0.7602.
      while (true) {
        _edist(_x, r);
        _edist(_z, r);
        for (mp_size_t k = 1; ; ++k) {
          _x.ExpandTo(r, k - 1);
          _z.ExpandTo(r, k - 1);
          mpfr_prec_t prec = (std::max)(mpfr_prec_t(MPFR_PREC_MIN), k * bits);
          mpfr_set_prec(_xf, prec);
          mpfr_set_prec(_zf, prec);
          // Try for acceptance first; so compute upper limit on (y-1)^2 and
          // lower limit on 2*z.
          if (_x.UInteger() == 0) {
            _x(_xf, MPFR_RNDD);
            mpfr_ui_sub(_xf, 1u, _xf, MPFR_RNDU);
          } else {
            _x(_xf, MPFR_RNDU);
            mpfr_sub_ui(_xf, _xf, 1u, MPFR_RNDU);
          }
          mpfr_sqr(_xf, _xf, MPFR_RNDU);
          _z(_zf, MPFR_RNDD);
          mpfr_mul_2ui(_zf, _zf, 1u, MPFR_RNDD);
          if (mpfr_cmp(_xf, _zf) < 0) {    // (y-1)^2 < 2*z, so accept
            if (_x.Boolean(r)) _x.Negate(); // include a random sign
            return;
          }
          // Try for rejection; so compute lower limit on (y-1)^2 and upper
          // limit on 2*z.
          if (_x.UInteger() == 0) {
            _x(_xf, MPFR_RNDU);
            mpfr_ui_sub(_xf, 1u, _xf, MPFR_RNDD);
          } else {
            _x(_xf, MPFR_RNDD);
            mpfr_sub_ui(_xf, _xf, 1u, MPFR_RNDD);
          }
          mpfr_sqr(_xf, _xf, MPFR_RNDD);
          _z(_zf, MPFR_RNDU);
          mpfr_mul_2ui(_zf, _zf, 1u, MPFR_RNDU);
          if (mpfr_cmp(_xf, _zf) > 0) // (y-1)^2 > 2*z, so reject
            break;
          // Otherwise repeat with more precision
        }
        // Reject and start over with a new y and z
      }
    }
    mutable MPFRRandom<bits> _x;
    mutable MPFRRandom<bits> _z;
    mutable mpfr_t _xf;
    mutable mpfr_t _zf;
    const MPFRExponential<bits> _edist;
  };

} // namespace RandomLib

#endif  // HAVE_MPFR
#endif  // RANDOMLIB_MPFRNORMALK_HPP
