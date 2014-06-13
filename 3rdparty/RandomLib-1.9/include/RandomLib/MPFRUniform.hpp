/**
 * \file MPFRUniform.hpp
 * \brief Header for MPFRUniform
 *
 * Sampling exactly from a uniform distribution for MPFR.
 *
 * Copyright (c) Charles Karney (2012) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_MPFRUNIFORM_HPP)
#define RANDOMLIB_MPFRUNIFORM_HPP 1

#include <RandomLib/MPFRRandom.hpp>

#if HAVE_MPFR || defined(DOXYGEN)

namespace RandomLib {

  /**
   * \brief The uniform distribution for MPFR.
   *
   * This is just a thin layer on top of MPFRRandom to provide random numbers
   * uniformly distributed in [0,1].
   *
   * This class uses a mutable private object.  So a single MPFRUniform object
   * cannot safely be used by multiple threads.  In a multi-processing
   * environment, each thread should use a thread-specific MPFRUniform object.
   *
   * @tparam bits the number of bits in each digit.
   **********************************************************************/
  template<int bits = 32> class MPFRUniform {
  public:

    /**
     * Initialize the MPFRUniform object.
     **********************************************************************/
    MPFRUniform() {};
    /**
     * Sample from the uniform distribution in [0,1] returning a MPFRRandom.
     * This function takes an unused GMP random generator as a parameter, in
     * order to parallel the usage of MPFRExponential and MPFRNormal.
     *
     * @param[out] t the MPFRRandom result.
     * @param[in,out] r a GMP random generator (unused).
     **********************************************************************/
    void operator()(MPFRRandom<bits>& t, gmp_randstate_t r) const
    { Compute(r); _x.swap(t); }
    /**
     * Sample from the uniform distribution in [0,1].
     *
     * @param[out] val the sample from the uniform distribution
     * @param[in,out] r a GMP random generator.
     * @param[in] round the rounding direction.
     * @return the MPFR ternary result (&plusmn; if val is larger/smaller than
     *   the exact sample).
     **********************************************************************/
    int operator()(mpfr_t val, gmp_randstate_t r, mpfr_rnd_t round) const
    { Compute(r); return _x(val, r, round); }
  private:
    // disable copy constructor and assignment operator
    MPFRUniform(const MPFRUniform&);
    MPFRUniform& operator=(const MPFRUniform&);
    void Compute(gmp_randstate_t /* r */) const { _x. Init(); }
    mutable MPFRRandom<bits> _x;
  };

} // namespace RandomLib

#endif  // HAVE_MPFR
#endif  // RANDOMLIB_MPFRUNIFORM_HPP
