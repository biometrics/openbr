/**
 * \file MPFRRandom.hpp
 * \brief Header for MPFRRandom
 *
 * Utility class for MPFRUniform, MPFRExponential, and MPFRNormal.
 *
 * Copyright (c) Charles Karney (2012) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_MPFRRANDOM_HPP)
#define RANDOMLIB_MPFRRANDOM_HPP 1

#include <algorithm>            // for swap
#include <mpfr.h>

#define HAVE_MPFR (MPFR_VERSION_MAJOR >= 3)

#if HAVE_MPFR || defined(DOXYGEN)

/**
 * A compile-time assert.  Use C++11 static_assert, if available.
 **********************************************************************/
#if !defined(STATIC_ASSERT)
#  if defined(__GXX_EXPERIMENTAL_CXX0X__)
#    define STATIC_ASSERT static_assert
#  elif defined(_MSC_VER) && _MSC_VER >= 1600
#    define STATIC_ASSERT static_assert
#  else
#    define STATIC_ASSERT(cond,reason) \
            { enum{ STATIC_ASSERT_ENUM = 1/int(cond) }; }
#  endif
#endif

namespace RandomLib {

  /**
   * \brief Handling random numbers in MPFR.
   *
   * This class provides roughly the same capabilities as RandomNumber.  The
   * fraction is represented by a mpz integer \e f and an exponent \e e.  We
   * have \e e &ge; 0 and 0 &le; \e f < <i>b</i><sup><i>e</i></sup>, and \e b =
   * 2<sup><i>bits</i></sup>.  This represents the number \e x = \e f
   * <i>b</i><sup>&minus;<i>e</i></sup>, with x in [0, 1).
   *
   * @tparam bits the number of bits in each digit.
   *
   * \e bits must divide GMP_LIMB_BITS.  The default value \e bits = 32 yields
   * portable results on all MPFR platforms.
   **********************************************************************/
  template<int bits = 32> class MPFRRandom {
  private:
    static const int limb_ = GMP_LIMB_BITS;  // How many bits in a limb
    static const int loglimb_ = (limb_ ==  32 ? 5 :
                                 (limb_ ==  64 ? 6 :
                                  (limb_ == 128 ? 7 : -1)));
    static const int logbits_ = (bits ==   1 ? 0 :
                                 (bits ==   2 ? 1 :
                                  (bits ==   4 ? 2 :
                                   (bits ==   8 ? 3 :
                                    (bits ==  16 ? 4 :
                                     (bits ==  32 ? 5 :
                                      (bits ==  64 ? 6 :
                                       (bits == 128 ? 7 :  -1))))))));
    static const mp_limb_t mask_ = (bits == limb_ ? ~0UL : // Digit mask
                                    ~(~0UL << (bits < limb_ ? bits : 0)));
    static const int logw_ = loglimb_ - logbits_; // 2^logw digits per limb
    static const unsigned wmask_ = ~(~0U << logw_);

    mutable mpz_t _tt;                                // A temporary
    mpz_t _f;                                         // The fraction
    mp_size_t _e;                                     // Count of digits
    unsigned long _n;                                 // Integer part
    int _s;                                           // Sign
    void AddDigits(gmp_randstate_t r, long num = 1) { // Add num more digits
      if (num <= 0) return;
      mpz_mul_2exp(_f, _f, num << logbits_);
      mpz_urandomb(_tt, r, num << logbits_);
      mpz_add(_f, _f, _tt);
      _e += num;
    }
    // return k'th digit counting k = 0 as most significant
    mp_limb_t Digit(gmp_randstate_t r, mp_size_t k) {
      ExpandTo(r, k);             // Now e > k
      k = _e - 1 - k;             // Reverse k so k = 0 is least significant
      // (k >> logw) is the limb index
      // (k & wmask) is the digit position within the limb
      return mask_ &
        (mpz_getlimbn(_f, k >> logw_) >> ((k & wmask_) << logbits_));
    }
    // Return index [0..32] of highest bit set.  Return 0 if x = 0, 32 is if x
    // = ~0.  (From Algorithms for programmers by Joerg Arndt.)
    static int highest_bit_idx(unsigned long x) throw() {
      if (x == 0) return 0;
      int r = 1;
      // STILL TO DO: handle 64-bit unsigned longs.
      if (x & 0xffff0000UL) { x >>= 16; r += 16; }
      if (x & 0x0000ff00UL) { x >>=  8; r +=  8; }
      if (x & 0x000000f0UL) { x >>=  4; r +=  4; }
      if (x & 0x0000000cUL) { x >>=  2; r +=  2; }
      if (x & 0x00000002UL) {           r +=  1; }
      return r;
    }
  public:
    /**
     * Initialize the MPFRRandom object.
     **********************************************************************/
    MPFRRandom() : _e(0u), _n(0u), _s(1) {
      STATIC_ASSERT(logbits_ >= 0 && loglimb_ >= 0 && logbits_ <= loglimb_,
                    "MPRFRandom: unsupported value for bits");
      mpz_init(_f); mpz_init(_tt);
    }
    /**
     * Initialize the MPFRRandom object from another one.
     *
     * @param[in] t the MPFRRandom to copy.
     **********************************************************************/
    MPFRRandom(const MPFRRandom& t) : _e(t._e), _n(t._n), _s(t._s)
    { mpz_init(_f); mpz_set(_f, t._f); mpz_init(_tt); }
    /**
     * Destroy the MPFRRandom object.
     **********************************************************************/
    ~MPFRRandom() { mpz_clear(_f); mpz_clear(_tt); }
    /**
     * Assignment operator.  (But swapping is typically faster.)
     *
     * @param[in] t the MPFRRandom to copy.
     **********************************************************************/
    MPFRRandom& operator=(const MPFRRandom& t) {
      _e = t._e;
      _n = t._n;
      _s = t._s;
      mpz_set(_f, t._f);        // Don't copy _tt
      return *this;
    }
    /**
     * Swap with another MPFRRandom.  This is a fast way of doing an
     * assignment.
     *
     * @param[in,out] t the MPFRRandom to swap with.
     **********************************************************************/
    void swap(MPFRRandom& t) throw() {
      if (this != &t) {
        std::swap(_e, t._e);
        std::swap(_n, t._n);
        std::swap(_s, t._s);
        mpz_swap(_f, t._f);     // Don't swap _tt
      }
    }
    /**
     * Reinitialize the MPFRRandom object, setting its value to [0,1].
     **********************************************************************/
    void Init() { mpz_set_ui(_f, 0u); _e = 0; _n = 0; _s = 1; }
    /**
     * @return the sign of the MPFRRandom (&plusmn; 1).
     **********************************************************************/
    int Sign() const throw() { return _s; }
    /**
     * Change the sign of the MPFRRandom.
     **********************************************************************/
    void Negate() throw() { _s *= -1; }
    /**
     * @return the floor of the MPFRRandom
     **********************************************************************/
    long Floor() const throw() { return _s > 0 ? long(_n) : -1 - long(_n); }
    /**
     * @return the ceiling of the MPFRRandom
     **********************************************************************/
    long Ceiling() const throw() { return _s > 0 ? 1 + long(_n) : -long(_n); }
    /**
     * @return the unsigned integer component of the MPFRRandom.
     **********************************************************************/
    unsigned long UInteger() const throw() { return _n; }
    /**
     * @return the number of digits in fraction
     **********************************************************************/
    unsigned long Size() const throw() { return unsigned(_e); }
    /**
     * Add integer \e k to the MPRFRandom.
     *
     * @param[in] k the integer to add.
     **********************************************************************/
    void AddInteger(long k) {
      k += Floor();             // The new floor
      int ns = k < 0 ? -1 : 1;  // The new sign
      if (ns != _s) {           // If sign changes, set f = 1 - f
        mpz_set_ui(_tt, 1u);
        mpz_mul_2exp(_tt, _tt, _e << logbits_);
        mpz_sub_ui(_tt, _tt, 1u);
        mpz_sub(_f, _tt, _f);
        _s = ns;
      }
      _n = ns > 0 ? k : -(k + 1);
    }
    /**
     * Compare with another MPFRRandom, *this < \e t.
     *
     * @param[in,out] r a random generator.
     * @param[in,out] t a MPFRRandom to compare.
     * @return true if *this < \e t.
     **********************************************************************/
    int LessThan(gmp_randstate_t r, MPFRRandom& t) {
      if (this == &t) return false; // same object
      if (_s != t._s) return _s < t._s;
      if (_n != t._n) return (_s < 0) ^ (_n < t._n);
      for (mp_size_t k = 0; ; ++k) {
        mp_limb_t x = Digit(r, k);
        mp_limb_t y = t.Digit(r, k);
        if (x != y) return (_s < 0) ^ (x < y);
      }
    }
    /**
     * Set high bit of fraction to 1.
     *
     * @param[in,out] r a random generator.
     **********************************************************************/
    void SetHighBit(gmp_randstate_t r) { // Set the msb to 1
      ExpandTo(r, 0);               // Generate msb if necessary
      mpz_setbit(_f, (_e << logbits_)  - 1);
    }
    /**
     * Test high bit of fraction.
     *
     * @param[in,out] r a random generator.
     **********************************************************************/
    int TestHighBit(gmp_randstate_t r) { // test the msb of f
      ExpandTo(r, 0);               // Generate msb if necessary
      return mpz_tstbit(_f, (_e << logbits_)  - 1);
    }
    /**
     * Return the position of the most significant bit in the MPFRRandom.
     *
     * @param[in,out] r a random generator.
     *
     * The bit position is numbered such the 1/2 bit is 0, the 1/4 bit is -1,
     * etc.
     **********************************************************************/
    mp_size_t LeadingBit(gmp_randstate_t r) {
      if (_n) return highest_bit_idx(_n);
      while (true) {
        int sgn = mpz_sgn(_f);
        if (sgn != 0)
          return mp_size_t(mpz_sizeinbase(_f, 2)) - mp_size_t(_e << logbits_);
        AddDigits(r);
      }
    }
    /**
     * Ensure that the k'th digit of the fraction is computed.
     *
     * @param[in,out] r a random generator.
     * @param[in] k the digit number (0 is the most significant, 1 is the next
     *   most significant, etc.
     **********************************************************************/
    void ExpandTo(gmp_randstate_t r, mp_size_t k)
    { if (_e <= k) AddDigits(r, k - _e + 1); }
    /**
     * Convert to a MPFR number \e without adding more bits.
     *
     * @param[out] val the value of s * (n + *this).
     * @param[in] round the rounding direction.
     * @return the MPFR ternary result (&plusmn; if val is larger/smaller than
     *   the exact sample).
     *
     * If round is MPFR_RNDN, then the rounded midpoint of the interval
     * represented by the MPFRRandom is returned.  Otherwise it is the rounded
     * lower or upper bound of the interval (whichever is appropriate).
     **********************************************************************/
    int operator()(mpfr_t val, mpfr_rnd_t round)
    { return operator()(val, NULL, round); }
    /**
     * Convert to a MPFR number.
     *
     * @param[out] val the value of s * (n + *this).
     * @param[in,out] r a GMP random generator.
     * @param[in] round the rounding direction.
     * @return the MPFR ternary result (&plusmn; if val is larger/smaller than
     *   the exact sample).
     *
     * If \e r is NULL, then no additional random bits are generated and the
     * lower bound, midpoint, or upper bound of the MPFRRandom interval is
     * returned, depending on the value of \e round.
     **********************************************************************/
    int operator()(mpfr_t val, gmp_randstate_t r, mpfr_rnd_t round) {
      // The value is constructed as a positive quantity, so adjust rounding
      // mode to account for this.
      switch (round) {
      case MPFR_RNDD:
      case MPFR_RNDU:
      case MPFR_RNDN:
        break;
      case MPFR_RNDZ:
        round = _s < 0 ? MPFR_RNDU : MPFR_RNDD;
        break;
      case MPFR_RNDA:
        round = _s < 0 ? MPFR_RNDD : MPFR_RNDU;
        break;
      default:
        round = MPFR_RNDN;      // New rounding modes are variants of N
        break;
      } // Now round is one of MPFR_RND{D,N,U}

      mp_size_t excess;
      mpfr_exp_t expt;
      if (r == NULL) {
        // If r is NULL then all the bits currently generated are considered
        // significant.  Thus no excess bits need to be squeezed out.
        excess = 0;
        // And the exponent shift in mpfr_set_z_2exp is just...
        expt = -(_e << logbits_);
        // However, if rounding to nearest, we need to make room for the
        // midpoint bit.
        if (round == MPFR_RNDN) {
          excess = -1;
          --expt;
        }
      } else {                  // r is non-NULL
        // Generate enough digits, i.e., enough to generate prec significant
        // figures for RNDD and RNDU; for RNDN we need to generate an
        // additional guard bit.
        mp_size_t lead = LeadingBit(r);
        mpfr_prec_t prec = mpfr_get_prec (val);
        mp_size_t trail = lead - prec; // position one past trailing bit
        mp_size_t guard = trail + (round == MPFR_RNDN ? 0 : 1); // guard bit pos
        // Generate the bits needed.
        if (guard <= 0) ExpandTo(r, (-guard) >> logbits_);
        // Unless bits = 1, the generation process will typically have
        // generated too many bits.  We figure out how many, but leaving room
        // for one additional "inexact" bit.  The inexact bit is set to 1 in
        // order to force MPFR to treat the result as inexact, to break RNDN
        // ties, and to get the ternary value set correctly.
        //
        // expt is the exponent used when forming the number using
        // mpfr_set_z_2exp.  Without the inexact bit, it's (guard - 1).
        // Subtract 1 to account for the inexact bit.
        expt = guard - 2;
        // The number of excess bits is now the difference between the number
        // of bits in the fraction (e << logbits) and -expt.  Note that this
        // may be -1 (meaning we'll need to shift the number left to
        // accommodate the inexact bit).
        excess = (_e << logbits_) + expt;
      }
      mpz_set_ui(_tt, _n);                    // The integer part
      mpz_mul_2exp(_tt, _tt, _e << logbits_); // Shift to allow for fraction
      mpz_add(_tt, _tt, _f);                  // Add fraction
      if (excess > 0)
        mpz_tdiv_q_2exp(_tt, _tt, excess);
      else if (excess < 0)
        mpz_mul_2exp(_tt, _tt, -excess);
      if (r || round == MPFR_RNDN)
        // Set the inexact bit (or compute the midpoint if r is NULL).
        mpz_setbit(_tt, 0);
      else if (round == MPFR_RNDU)
        // If r is NULL, compute the upper bound.
        mpz_add_ui(_tt, _tt, 1u);

      // Convert to a mpfr number.  If r is specified, then there are
      // sufficient bits in tt that the result is inexact and that (in the case
      // of RNDN) there are no ties.
      int flag = mpfr_set_z_2exp(val, _tt, expt, round);
      if (_s < 0) {
        mpfr_neg (val, val, MPFR_RNDN);
        flag = -flag;
      }
      return flag;
    }
    /**
     * A coin toss.  (This should really be a static function.  But it uses the
     * MPFRRandom temporary variable.)
     *
     * @param[in,out] r a GMP random generator.
     * @return true or false.
     **********************************************************************/
    int Boolean(gmp_randstate_t r) const {
      mpz_urandomb(_tt, r, 1);
      return mpz_tstbit(_tt, 0);
    }
  };

} // namespace RandomLib

#endif  // HAVE_MPFR
#endif  // RANDOMLIB_MPFRRANDOM_HPP
