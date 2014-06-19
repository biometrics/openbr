/**
 * \file RandomPower2.hpp
 * \brief Header for RandomPower2.
 *
 * Return and multiply by powers of two.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_RANDOMPOWER2_HPP)
#define RANDOMLIB_RANDOMPOWER2_HPP 1

#include <cmath>                // For std::pow

namespace RandomLib {

  /**
   * \brief Return or multiply by powers of 2
   *
   * With some compilers it's fastest to do a table lookup of powers of
   * 2.  If RANDOMLIB_POWERTABLE is 1, a lookup table is used.  If
   * RANDOMLIB_POWERTABLE is 0, then std::pow is used.
   **********************************************************************/
  class RANDOMLIB_EXPORT RandomPower2 {
  public:
    /**
     * Return powers of 2 (either using a lookup table or std::pow)
     *
     * @param[in] n the integer power.
     * @return 2<sup><i>n</i></sup>.
     **********************************************************************/
    template<typename RealType> static inline RealType pow2(int n) throw() {
#if RANDOMLIB_POWERTABLE
      return RealType(power2[n - minpow]);
#else
      return std::pow(RealType(2), n);
#endif
    }
    /**
     * Multiply a real by a power of 2
     *
     * @tparam RealType the type of \e x.
     * @param[in] x the real number.
     * @param[in] n the power (positive or negative).
     * @return \e x 2<sup><i>n</i></sup>.
     **********************************************************************/
    template<typename RealType>
    static inline RealType shiftf(RealType x, int n) throw()
    // std::ldexp(x, n); is equivalent, but slower
    { return x * pow2<RealType>(n); }

    // Constants
    enum {
      /**
       * Minimum power in RandomPower2::power2
       **********************************************************************/
#if RANDOMLIB_LONGDOUBLEPREC > 64
      minpow = -120,
#else
      minpow = -64,
#endif
      maxpow = 64               /**< Maximum power in RandomPower2::power2. */
    };
  private:
#if RANDOMLIB_POWERTABLE
    /**
     * Table of powers of two
     **********************************************************************/
    static const float power2[maxpow - minpow + 1]; // Powers of two
#endif
  };

} // namespace RandomLib

#endif  // RANDOMLIB_RANDOMPOWER2_HPP
