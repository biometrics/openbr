/**
 * \file MPFRNormal.hpp
 * \brief Header for MPFRNormal
 *
 * Sampling exactly from the normal distribution for MPFR.
 *
 * Copyright (c) Charles Karney (2012) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_MPFRNORMAL_HPP)
#define RANDOMLIB_MPFRNORMAL_HPP 1

#include <algorithm>            // for max/min
#include <RandomLib/MPFRRandom.hpp>

#if HAVE_MPFR || defined(DOXYGEN)

namespace RandomLib {

  /**
   * \brief The normal distribution for MPFR.
   *
   * This is a transcription of ExactNormal (version 1.3) for use with MPFR.
   *
   * This class uses mutable private objects.  So a single MPFRNormal object
   * cannot safely be used by multiple threads.  In a multi-processing
   * environment, each thread should use a thread-specific MPFRNormal object.
   *
   * @tparam bits the number of bits in each digit.
   **********************************************************************/
  template<int bits = 32> class MPFRNormal {
  public:

    /**
     * Initialize the MPFRNormal object.
     **********************************************************************/
    MPFRNormal() { mpz_init(_tt); }
    /**
     * Destroy the MPFRNormal object.
     **********************************************************************/
    ~MPFRNormal() { mpz_clear(_tt); }
    /**
     * Sample from the normal distribution with mean 0 and variance 1 returning
     * a MPFRRandom.
     *
     * @param[out] t the MPFRRandom result.
     * @param[in,out] r a GMP random generator.
     **********************************************************************/
    void operator()(MPFRRandom<bits>& t,gmp_randstate_t r) const
    { Compute(r); return _x.swap(t); }
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
    // Disable copy constructor and assignment operator
    MPFRNormal(const MPFRNormal&);
    MPFRNormal& operator=(const MPFRNormal&);
    // True with prob exp(-1/2)
    int ExpProbH(gmp_randstate_t r) const {
      _p.Init(); if (_p.TestHighBit(r)) return 1;
      // von Neumann rejection
      while (true) {
        _q.Init(); if (!_q.LessThan(r, _p)) return 0;
        _p.Init(); if (!_p.LessThan(r, _q)) return 1;
      }
    }
    // True with prob exp(-n/2)
    int ExpProb(gmp_randstate_t r, unsigned n) const {
      while (n--) { if (!ExpProbH(r)) return 0; }
      return 1;
    }
    // n with prob (1-exp(-1/2)) * exp(-n/2)
    unsigned ExpProbN(gmp_randstate_t r) const {
      unsigned n = 0;
      while (ExpProbH(r)) ++n;
      return n;
    }
    // Return:
    //  1 with prob 2k/(2k + 2)
    //  0 with prob  1/(2k + 2)
    // -1 with prob  1/(2k + 2)
    int Choose(gmp_randstate_t r, int k) const {
      const int b = 15;           // To avoid integer overflow on multiplication
      const int m = 2 * k + 2;
      int n1 = m - 2, n2 = m - 1;
      while (true) {
        mpz_urandomb(_tt, r, b);
        int d = int( mpz_get_ui(_tt) ) * m;
        n1 = (std::max)((n1 << b) - d, 0);
        if (n1 >= m) return 1;
        n2 = (std::min)((n2 << b) - d, m);
        if (n2 <= 0) return -1;
        if (n1 == 0 && n2 == m) return 0;
      }
    }
    void Compute(gmp_randstate_t r) const {
      while (true) {
        unsigned k = ExpProbN(r); // the integer part of the result.
        if (ExpProb(r, (k - 1) * k)) {
          _x.Init();
          unsigned s = 1;
          for (unsigned j = 0; j <= k; ++j) { // execute k + 1 times
            bool first;
            for (s = 1, first = true; ; s ^= 1, first = false) {
              if (k == 0 && _x.Boolean(r)) break;
              _q.Init(); if (!_q.LessThan(r, first ? _x : _p)) break;
              int y = k == 0 ? 0 : Choose(r, k);
              if (y < 0)
                break;
              else if (y == 0) {
                _p.Init(); if (!_p.LessThan(r, _x)) break;
              }
              _p.swap(_q);        // a fast way of doing p = q
            }
            if (s == 0) break;
          }
          if (s != 0) {
            _x.AddInteger(k);
            if (_x.Boolean(r)) _x.Negate();
            return;
          }
        }
      }
    }
    mutable mpz_t _tt;          // A temporary
    mutable MPFRRandom<bits> _x;
    mutable MPFRRandom<bits> _p;
    mutable MPFRRandom<bits> _q;
  };

} // namespace RandomLib

#endif  // HAVE_MPFR
#endif  // RANDOMLIB_MPFRNORMAL_HPP
