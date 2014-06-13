/**
 * \file MPFRExponential.hpp
 * \brief Header for MPFRExponential
 *
 * Sampling exactly from the normal distribution for MPFR.
 *
 * Copyright (c) Charles Karney (2012) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_MPFREXPONENTIAL_HPP)
#define RANDOMLIB_MPFREXPONENTIAL_HPP 1

#include <RandomLib/MPFRRandom.hpp>

#if HAVE_MPFR || defined(DOXYGEN)

namespace RandomLib {

  /**
   * \brief The exponential distribution for MPFR.
   *
   * This is a transcription of ExactExponential (version 1.4) for use with
   * MPFR.
   *
   * This class uses mutable private objects.  So a single MPFRExponential
   * object cannot safely be used by multiple threads.  In a multi-processing
   * environment, each thread should use a thread-specific MPFRExponential
   * object.
   *
   * @tparam bits the number of bits in each digit.
   **********************************************************************/
  template<int bits = 32> class MPFRExponential {
  public:

    /**
     * Initialize the MPFRExponential object.
     **********************************************************************/
    MPFRExponential() {};
    /**
     * Sample from the exponential distribution with mean 1 returning a
     * MPFRRandom.
     *
     * @param[out] t the MPFRRandom result.
     * @param[in,out] r a GMP random generator.
     **********************************************************************/
    void operator()(MPFRRandom<bits>& t, gmp_randstate_t r) const
    { Compute(r); _x.swap(t); }
    /**
     * Sample from the exponential distribution with mean 1.
     *
     * @param[out] val the sample from the exponential distribution
     * @param[in,out] r a GMP random generator.
     * @param[in] round the rounding direction.
     * @return the MPFR ternary result (&plusmn; if val is larger/smaller than
     *   the exact sample).
     **********************************************************************/
    int operator()(mpfr_t val, gmp_randstate_t r, mpfr_rnd_t round) const
    { Compute(r); return _x(val, r, round); }
  private:
    // Disable copy constructor and assignment operator
    MPFRExponential(const MPFRExponential&);
    MPFRExponential& operator=(const MPFRExponential&);
    int ExpFraction(gmp_randstate_t r, MPFRRandom<bits>& p) const {
      // The early bale out
      if (p.TestHighBit(r)) return 0;
      // Implement the von Neumann algorithm
      _w.Init();
      if (!_w.LessThan(r, p)) return 1;
      while (true) {
        _v.Init(); if (!_v.LessThan(r, _w)) return 0;
        _w.Init(); if (!_w.LessThan(r, _v)) return 1;
      }
    }
    void Compute(gmp_randstate_t r) const {
      _x.Init();
      unsigned k = 0;
      while (!ExpFraction(r, _x)) { ++k; _x.Init(); }
      if (k & 1) _x.SetHighBit(r);
      _x.AddInteger(k >> 1);
      return;
    }
    mutable MPFRRandom<bits> _x;
    mutable MPFRRandom<bits> _v;
    mutable MPFRRandom<bits> _w;
  };

} // namespace RandomLib

#endif  // HAVE_MPFR
#endif  // RANDOMLIB_MPFREXPONENTIAL_HPP
