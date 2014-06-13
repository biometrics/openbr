/**
 * \file MPFRNormalR.hpp
 * \brief Header for MPFRNormalR
 *
 * Sampling exactly from the normal distribution for MPFR using the ratio
 * method.
 *
 * Copyright (c) Charles Karney (2012) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_MPFRNORMALR_HPP)
#define RANDOMLIB_MPFRNORMALR_HPP 1

#include <algorithm>            // for max/min
#include <cmath>                // for pow
#include <mpfr.h>

#define HAVE_MPFR (MPFR_VERSION_MAJOR >= 3)

#if HAVE_MPFR || defined(DOXYGEN)

namespace RandomLib {

  /**
   * \brief The normal distribution for MPFR (ratio method).
   *
   * This class is <b>DEPRECATED</b>.  It is included for illustrative purposes
   * only.  The MPFRNormal class provides a much more efficient method for
   * sampling from the normal distribution.
   *
   * This is an adaption of NormalDistribution to MPFR.  The changes are
   * - Use MPFR's random number generator
   * - Use sufficient precision internally to ensure that a correctly rounded
   *   result is returned.
   *
   * This class uses a mutable private object.  So a single MPFRNormalR
   * object cannot safely be used by multiple threads.  In a multi-processing
   * environment, each thread should use a thread-specific MPFRNormalR
   * object.
   **********************************************************************/
  class MPFRNormalR {
  private:
    // The number of bits of randomness to add at a time.  Require that Leva's
    // bounds "work" at a precision of 2^-chunk and that an unsigned long can
    // hold this many bits.
    static const long chunk_ = 32;
    static const unsigned long m = 3684067834; // ceil(2^chunk*sqrt(2/e))

  public:
    /**
     * Initialize the MPFRNormalR object.
     **********************************************************************/
    MPFRNormalR() {
      mpz_init(_ui);
      mpz_init(_vi);
      mpfr_init2(_eps, chunk_);
      mpfr_init2(_u, chunk_);
      mpfr_init2(_v, chunk_);
      mpfr_init2(_up, chunk_);
      mpfr_init2(_vp, chunk_);
      mpfr_init2(_vx, chunk_);
      mpfr_init2(_x1, chunk_);
      mpfr_init2(_x2, chunk_);
    }
    /**
     * Destroy the MPFRNormalR object.
     **********************************************************************/
    ~MPFRNormalR() {
      mpfr_clear(_x2);
      mpfr_clear(_x1);
      mpfr_clear(_vx);
      mpfr_clear(_vp);
      mpfr_clear(_up);
      mpfr_clear(_v);
      mpfr_clear(_u);
      mpfr_clear(_eps);
      mpz_clear(_vi);
      mpz_clear(_ui);
    }
    /**
     * Sample from the normal distribution with mean 0 and variance 1.
     *
     * @param[out] val the sample from the normal distribution
     * @param[in,out] r a GMP random generator.
     * @param[in] round the rounding direction.
     * @return the MPFR ternary result (&plusmn;1 if val is larger/smaller than
     *   the exact sample).
     **********************************************************************/
    int operator()(mpfr_t val, gmp_randstate_t r, mpfr_rnd_t round) const {
      const double
        s  =  0.449871, // Constants from Leva
        t  = -0.386595,
        a  =  0.19600 ,
        b  =  0.25472 ,
        r1 =  0.27597 ,
        r2 =  0.27846 ,
        u1 =  0.606530,           // sqrt(1/e) rounded down and up
        u2 =  0.606531,
        scale = std::pow(2.0, -chunk_); // for turning randoms into doubles

      while (true) {
        mpz_urandomb(_vi, r, chunk_);
        if (mpz_cmp_ui(_vi, m) >= 0) continue; // Very early reject
        double vf = (mpz_get_ui(_vi) + 0.5) * scale;
        mpz_urandomb(_ui, r, chunk_);
        double uf = (mpz_get_ui(_ui) + 0.5) * scale;
        double
          x = uf - s,
          y = vf - t,
          Q = x*x + y * (a*y - b*x);
        if (Q >= r2) continue;    // Early reject
        mpfr_set_ui_2exp(_eps, 1u, -chunk_, MPFR_RNDN);
        mpfr_prec_t prec = chunk_;
        mpfr_set_prec(_u, prec);
        mpfr_set_prec(_v, prec);
        // (u,v) = sw corner of range
        mpfr_set_z_2exp(_u, _ui, -prec, MPFR_RNDN);
        mpfr_set_z_2exp(_v, _vi, -prec, MPFR_RNDN);
        mpfr_set_prec(_up, prec);
        mpfr_set_prec(_vp, prec);
        // (up,vp) = ne corner of range
        mpfr_add(_up, _u, _eps, MPFR_RNDN);
        mpfr_add(_vp, _v, _eps, MPFR_RNDN);
        // Estimate how many extra bits will be needed to achieve the desired
        // precision.
        mpfr_prec_t prec_guard = 3 + chunk_ -
          (std::max)(mpz_sizeinbase(_ui, 2), mpz_sizeinbase(_vi, 2));
        if (Q > r1) {
          int reject;
          while (true) {
            // Rejection curve v^2 + 4 * u^2 * log(u) < 0 has a peak at u =
            // exp(-1/2) = 0.60653066.  So treat uf in (0.606530, 0.606531) =
            // (u1, u2) specially

            // Try for rejection first
            if (uf <= u1)
              reject = Reject(_u, _vp, prec, MPFR_RNDU);
            else if (uf >= u2)
              reject = Reject(_up, _vp, prec, MPFR_RNDU);
            else {              // u in (u1, u2)
              mpfr_set_prec(_vx, prec);
              mpfr_add(_vx, _vp, _eps, MPFR_RNDN);
              reject = Reject(_u, _vx, prec, MPFR_RNDU); // Could use _up too
            }
            if (reject < 0) break; // tried to reject but failed, so accept

            // Try for acceptance
            if (uf <= u1)
              reject = Reject(_up, _v, prec, MPFR_RNDD);
            else if (uf >= u2)
              reject = Reject(_u, _v, prec, MPFR_RNDD);
            else {              // u in (u2, u2)
              mpfr_sub(_vx, _v, _eps, MPFR_RNDN);
              reject = Reject(_u, _vx, prec, MPFR_RNDD); // Could use _up too
            }
            if (reject > 0) break; // tried to accept but failed, so reject

            prec = Refine(r, prec);  // still can't decide, to refine
          }
          if (reject > 0) continue; // reject, back to outer loop
        }
        // Now evaluate v/u to the necessary precision
        mpfr_prec_t prec0 = mpfr_get_prec (val);
        //        while (prec < prec0 + prec_guard) prec = Refine(r, prec);
        if (prec < prec0 + prec_guard)
          prec = Refine(r, prec,
                        (prec0 + prec_guard - prec + chunk_ - 1) / chunk_);
        mpfr_set_prec(_x1, prec0);
        mpfr_set_prec(_x2, prec0);
        int flag;
        while (true) {
          int
            f1 = mpfr_div(_x1, _v, _up, round),   // min slope
            f2 = mpfr_div(_x2, _vp, _u, round);   // max slope
          if (f1 == f2 && mpfr_equal_p(_x1, _x2)) {
            flag = f1;
            break;
          }
          prec = Refine(r, prec);
        }
        mpz_urandomb(_ui, r, 1);
        if (mpz_tstbit(_ui, 0)) {
          flag = -flag;
          mpfr_neg(val, _x1, MPFR_RNDN);
        } else
          mpfr_set(val, _x1, MPFR_RNDN);
        //      std::cerr << uf << " " << vf << " " << Q << "\n";
        return flag;
      }
    }
  private:
    // disable copy constructor and assignment operator
    MPFRNormalR(const MPFRNormalR&);
    MPFRNormalR& operator=(const MPFRNormalR&);
    // Refine the random square
    mpfr_prec_t Refine(gmp_randstate_t r, mpfr_prec_t prec, long num = 1)
      const {
      if (num <= 0) return prec;
      // Use _vx as scratch
      prec += num * chunk_;
      mpfr_div_2ui(_eps, _eps, num * chunk_, MPFR_RNDN);

      mpz_urandomb(_ui, r, num * chunk_);
      mpfr_set_prec(_up, prec);
      mpfr_set_z_2exp(_up, _ui, -prec, MPFR_RNDN);
      mpfr_set_prec(_vx, prec);
      mpfr_add(_vx, _u, _up, MPFR_RNDN);
      mpfr_swap(_u, _vx);       // u = vx
      mpfr_add(_up, _u, _eps, MPFR_RNDN);

      mpz_urandomb(_vi, r, num * chunk_);
      mpfr_set_prec(_vp, prec);
      mpfr_set_z_2exp(_vp, _vi, -prec, MPFR_RNDN);
      mpfr_set_prec(_vx, prec);
      mpfr_add(_vx, _v, _vp, MPFR_RNDN);
      mpfr_swap(_v, _vx);       // v = vx
      mpfr_add(_vp, _v, _eps, MPFR_RNDN);

      return prec;
    }
    // Evaluate the sign of the rejection condition v^2 + 4*u^2*log(u)
    int Reject(mpfr_t u, mpfr_t v, mpfr_prec_t prec, mpfr_rnd_t round) const {
      // Use x1, x2 as scratch
      mpfr_set_prec(_x1, prec);

      mpfr_log(_x1, u, round);
      mpfr_mul(_x1, _x1, u, round); // Important to do the multiplications in
      mpfr_mul(_x1, _x1, u, round); // this order so that rounding works right.
      mpfr_mul_2ui(_x1, _x1, 2u, round); // 4*u^2*log(u)

      mpfr_set_prec(_x2, prec);
      mpfr_mul(_x2, v, v, round);        // v^2

      mpfr_add(_x1, _x1, _x2, round);    // v^2 + 4*u^2*log(u)

      return mpfr_sgn(_x1);
    }
    mutable mpz_t _ui;
    mutable mpz_t _vi;
    mutable mpfr_t _eps;
    mutable mpfr_t _u;
    mutable mpfr_t _v;
    mutable mpfr_t _up;
    mutable mpfr_t _vp;
    mutable mpfr_t _vx;
    mutable mpfr_t _x1;
    mutable mpfr_t _x2;
  };

} // namespace RandomLib

#endif  // HAVE_MPFR
#endif  // RANDOMLIB_MPFRNORMALR_HPP
