/**
 * \file MPFRExponentialL.hpp
 * \brief Header for MPFRExponentialL
 *
 * Sampling exactly from the exponential distribution for MPFR using the
 * traditional method.
 *
 * Copyright (c) Charles Karney (2012) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_MPFREXPONENTIALL_HPP)
#define RANDOMLIB_MPFREXPONENTIALL_HPP 1

#include <cmath>                // for log
#include <mpfr.h>

#define HAVE_MPFR (MPFR_VERSION_MAJOR >= 3)

#if HAVE_MPFR || defined(DOXYGEN)

namespace RandomLib {

  /**
   * \brief The exponential distribution for MPFR (the log method).
   *
   * This class is <b>DEPRECATED</b>.  It is included for illustrative purposes
   * only.  The MPFRExponential class provides a much more efficient method for
   * sampling from the exponential distribution.
   *
   * This is an adaption of ExponentialDistribution to MPFR.  The changes are
   * - Use MPFR's random number generator
   * - Use sufficient precision internally to ensure that a correctly rounded
   *   result is returned.
   *
   * This class uses mutable private objects.  So a single MPFRExponentialL
   * object cannot safely be used by multiple threads.  In a multi-processing
   * environment, each thread should use a thread-specific MPFRExponentialL
   * object.
   **********************************************************************/
  class MPFRExponentialL {
  private:
    // The number of bits of randomness to add at a time.
    static const long chunk_ = 32;

  public:
    /**
     * Initialize the MPFRExponentialL object.
     **********************************************************************/
    MPFRExponentialL() {
      mpz_init(_vi);
      mpfr_init2(_eps, chunk_);
      mpfr_init2(_v1, chunk_);
      mpfr_init2(_v2, chunk_);
    }
    /**
     * Destroy the MPFRExponentialL object.
     **********************************************************************/
    ~MPFRExponentialL() {
      mpfr_clear(_v2);
      mpfr_clear(_v1);
      mpfr_clear(_eps);
      mpz_clear(_vi);
    }
    /**
     * Sample from the exponential distribution with mean 1.
     *
     * @param[out] val the sample from the exponential distribution
     * @param[in,out] r a GMP random generator.
     * @param[in] round the rounding direction.
     * @return the MPFR ternary result (&plusmn; if val is larger/smaller than
     *   the exact sample).
     **********************************************************************/
    int operator()(mpfr_t val, gmp_randstate_t r, mpfr_rnd_t round) const {

      mpfr_prec_t prec0 = mpfr_get_prec (val);
      mpfr_prec_t prec = prec0 + 10; // A rough optimum
      mpz_urandomb(_vi, r, prec);
      mpfr_set_ui_2exp(_eps, 1u, -prec, MPFR_RNDN);
      mpfr_set_prec(_v1, prec);
      mpfr_set_z_2exp(_v1, _vi, -prec, MPFR_RNDN);
      mpfr_set_prec(_v2, prec);
      mpfr_add(_v2, _v1, _eps, MPFR_RNDN);
      while (true) {
        int f2 = mpfr_log(val, _v2, round); // val = log(upper bound)
        mpfr_set_prec(_v2, prec0);
        int f1 = mpfr_log(_v2, _v1, round); // v2 = log(lower bound)
        if (f1 == f2 && mpfr_equal_p(val, _v2)) {
          mpfr_neg(val, val, MPFR_RNDN);
          return -f1;
        }
        prec = Refine(r, prec);
      }
    }
  private:
    // disable copy constructor and assignment operator
    MPFRExponentialL(const MPFRExponentialL&);
    MPFRExponentialL& operator=(const MPFRExponentialL&);
    // Refine the random interval
    mpfr_prec_t Refine(gmp_randstate_t r, mpfr_prec_t prec)
      const {
      prec += chunk_;
      mpfr_div_2ui(_eps, _eps, chunk_, MPFR_RNDN);
      mpz_urandomb(_vi, r, chunk_);
      mpfr_set_prec(_v2, prec);
      mpfr_set_z_2exp(_v2, _vi, -prec, MPFR_RNDN);
      mpfr_add(_v2, _v1, _v2, MPFR_RNDN);
      mpfr_swap(_v1, _v2);      // v1 = v2;
      mpfr_set_prec(_v2, prec);
      mpfr_add(_v2, _v1, _eps, MPFR_RNDN);
      return prec;
    }
    mutable mpz_t _vi;
    mutable mpfr_t _eps;
    mutable mpfr_t _v1;
    mutable mpfr_t _v2;
  };

} // namespace RandomLib

#endif  // HAVE_MPFR
#endif  // RANDOMLIB_MPFREXPONENTIALL_HPP
