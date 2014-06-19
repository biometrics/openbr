/**
 * \file UniformInteger.hpp
 * \brief Header for UniformInteger
 *
 * Partially sample a uniform integer distribution.
 *
 * Copyright (c) Charles Karney (2013) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_UNIFORMINTEGER_HPP)
#define RANDOMLIB_UNIFORMINTEGER_HPP 1

#include <limits>

namespace RandomLib {
  /**
   * \brief The partial uniform integer distribution.
   *
   * A class to sample in [0, \e m).  For background, see:
   * - D. E. Knuth and A. C. Yao, The Complexity of Nonuniform Random Number
   *   Generation, in "Algorithms and Complexity" (Academic Press, 1976),
   *   pp. 357--428.
   * - J. Lumbroso, Optimal Discrete Uniform Generation from Coin Flips,
   *   and Applications, http://arxiv.org/abs/1304.1916 (2013)
   * .
   * Lumbroso's algorithm is a realization of the Knuth-Yao method for the case
   * of uniform probabilities.  This class generalizes the method to accept
   * random digits in a base, \e b = 2<sup>\e bits</sup>.  An important
   * additional feature is that only sufficient random digits are drawn to
   * narrow the allowed range to a power of b.  Thus after
   * <code>UniformInteger<int,1> u(r,5)</code>, \e u represents \verbatim
      range prob
      [0,4) 8/15
      [0,2) 2/15
      [2,4) 2/15
      4     1/5 \endverbatim
   * <code>u.Min()</code> and <code>u.Max()</code> give the extent of the
   * closed range.  The number of additional random digits needed to fix the
   * value is given by <code>u.Entropy()</code>.  The comparison operations may
   * require additional digits to be drawn and so the range might be narrowed
   * down.  If you need a definite value then use <code>u(r)</code>.
   *
   * The DiscreteNormalAlt class uses UniformInteger to achieve an
   * asymptotically ideal scaling wherein the number of random bits required
   * per sample is constant + log<sub>2</sub>&sigma;.  If Lumbroso's algorithm
   * for sampling in [0,\e m) were used the log<sub>2</sub>&sigma; term would
   * be multiplied by about 1.4.
   *
   * It is instructive to look at the Knuth-Yao discrete distribution
   * generating (DDG) tree for the case \e m = 5 (the binary expansion of 1/5
   * is 0.00110011...); Lumbroso's algorithm implements this tree.
   * \image html ky-5.png "Knuth-Yao for \e m = 5"
   *
   * UniformInteger collapses all of the full subtrees above to their parent
   * nodes to yield this tree where now some of the outcomes are ranges.
   * \image html ky-5-collapse.png "Collapsed Knuth-Yao for \e m = 5"
   *
   * Averaging over many samples, the maximum number of digits required to
   * construct a UniformInteger, i.e., invoking
   * <code>UniformInteger(r,m)</code>, is (2\e b &minus; 1)/(\e b &minus; 1).
   * (Note that this does not increase as \e m increases.)  The maximum number
   * of digits required to sample specific integers, i.e., invoking
   * <code>UniformInteger(r,m)(r)</code>, is <i>b</i>/(\e b &minus; 1) +
   * log<sub>\e b</sub>\e m.  The worst cases are when \e m is slightly more
   * than a power of \e b.
   *
   * The number of random bits required for sampling is shown as a function of
   * the fractional part of log<sub>2</sub>\e m below.  The red line shows what
   * Lumbroso calls the "toll", the number of bits in excess of the entropy
   * that are required for sampling.
   * \image html
   * uniform-bits.png "Random bits to sample in [0,\e m) for \e b = 2"
   *
   * @tparam IntType the type of the integer (must be signed).
   * @tparam bits the number of bits in each digit used for sampling;
   *   the base for sampling is \e b = 2<sup>\e bits</sup>.
   **********************************************************************/
  template<typename IntType = int, int bits = 1>  class UniformInteger {
  public:
    /**
     * Constructor creating a partially sampled integer in [0, \e m)
     *
     * @param[in] r random object.
     * @param[in] m constructed object represents an integer in [0, \e m).
     * @param[in] flip if true, rearrange the ranges so that the widest ones
     *   are at near the upper end of [0, \e m) (default false).
     *
     * The samples enough random digits to obtain a uniform range whose size is
     * a power of the base.  The range can subsequently be narrowed by sampling
     * additional digits.
     **********************************************************************/
    template<class Random>
    UniformInteger(Random& r, IntType m, bool flip = false);
    /**
     * @return the minimum of the current range.
     **********************************************************************/
    IntType Min() const { return _a; }
    /**
     * @return the maximum of the current range.
     **********************************************************************/
    IntType Max() const { return _a + (IntType(1) << (_l * bits)) - 1; }
    /**
     * @return the entropy of the current range (in units of random digits).
     *
     * Max() + 1 - Min() = 2<sup>Entropy() * \e bits</sup>.
     **********************************************************************/
    IntType Entropy() const { return _l; }
    /**
     * Sample until the entropy vanishes, i.e., Min() = Max().
     *
     * @return the resulting integer sample.
     **********************************************************************/
    template<class Random> IntType operator()(Random& r)
    { while (_l) Refine(r); return _a; }
    /**
     * Negate the range, [Min(), Max()] &rarr; [&minus;Max(), &minus;Min()].
     **********************************************************************/
    void Negate() { _a = -Max(); }
    /**
     * Add a constant to the range
     *
     * @param[in] c the constant to be added.
     *
     * [Min(), Max()] &rarr; [Min() + \e c, Max() + \e c].
     **********************************************************************/
    void Add(IntType c) { _a += c; }
    /**
     * Compare with a fraction, *this &lt; <i>p</i>/<i>q</i>
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in] p the numerator of the fraction.
     * @param[in] q the denominator of the fraction (require \e q &gt; 0).
     * @return true if *this &lt; <i>p</i>/<i>q</i>.
     **********************************************************************/
    // test j < p/q (require q > 0)
    template<class Random> bool LessThan(Random& r, IntType p, IntType q) {
      for (;;) {
        if ( (q * Max() < p)) return true;
        if (!(q * Min() < p)) return false;
        Refine(r);
      }
    }
    /**
     * Compare with a fraction, *this &le; <i>p</i>/<i>q</i>
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in] p the numerator of the fraction.
     * @param[in] q the denominator of the fraction (require \e q &gt; 0).
     * @return true if *this &le; <i>p</i>/<i>q</i>.
     **********************************************************************/
    template<class Random>
    bool LessThanEqual(Random& r, IntType p, IntType q)
    { return LessThan(r, p + 1, q); }
    /**
     * Compare with a fraction, *this &gt; <i>p</i>/<i>q</i>
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in] p the numerator of the fraction.
     * @param[in] q the denominator of the fraction (require \e q &gt; 0).
     * @return true if *this &gt; <i>p</i>/<i>q</i>.
     **********************************************************************/
    template<class Random>
    bool GreaterThan(Random& r, IntType p, IntType q)
    { return !LessThanEqual(r, p, q); }
    /**
     * Compare with a fraction, *this &ge; <i>p</i>/<i>q</i>
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in] p the numerator of the fraction.
     * @param[in] q the denominator of the fraction (require \e q &gt; 0).
     * @return true if *this &ge; <i>p</i>/<i>q</i>.
     **********************************************************************/
    template<class Random>
    bool GreaterThanEqual(Random& r, IntType p, IntType q)
    { return !LessThan(r, p, q); }
    /**
     * Check that overflow will not happen.
     *
     * @param[in] mmax the largest \e m in the constructor.
     * @param[in] qmax the largest \e q in LessThan().
     * @return true if overflow will not happen.
     *
     * It is important that this check be carried out.  If overflow occurs,
     * incorrect results are obtained and the constructor may never terminate.
     **********************************************************************/
    static bool Check(IntType mmax, IntType qmax) {
      return ( mmax - 1 <= ((std::numeric_limits<IntType>::max)() >> bits) &&
               mmax - 1 <= (std::numeric_limits<IntType>::max)() / qmax );
    }
  private:
    IntType _a, _l;             // current range is _a + [0, 2^(bits*_l)).
    template<class Random> static unsigned RandomDigit(Random& r) throw()
    { return unsigned(r.template Integer<bits>()); }
    template<class Random> void Refine(Random& r) // only gets called if _l > 0.
    { _a += IntType(RandomDigit(r) << (bits * --_l)); }
  };

  template<typename IntType, int bits> template<class Random>
  UniformInteger<IntType, bits>::UniformInteger(Random& r, IntType m, bool flip)
  {
    STATIC_ASSERT(std::numeric_limits<IntType>::is_integer,
                  "UniformInteger: invalid integer type IntType");
    STATIC_ASSERT(std::numeric_limits<IntType>::is_signed,
                  "UniformInteger: IntType must be a signed type");
    STATIC_ASSERT(bits > 0 && bits < std::numeric_limits<IntType>::digits &&
                  bits <= std::numeric_limits<unsigned>::digits,
                  "UniformInteger: bits out of range");
    m = m < 1 ? 1 : m;
    for (IntType v = 1, c = 0;;) {
      _l = 0; _a = c;
      for (IntType w = v, a = c, d = 1;;) {
        // play out Lumbroso's algorithm without drawing random digits with w
        // playing the role of v and c represented by the range [a, a + d).
        // Return if both ends of range qualify as return values at the same
        // time.  Otherwise, fail and draw another random digit.
        if (w >= m) {
          IntType j = (a / m) * m; a -= j; w -= j;
          if (w >= m) {
            if (a + d <= m) { _a = !flip ? a : m - a - d; return; }
            goto draw;
          }
        }
        w <<= bits; a <<= bits; d <<= bits; ++_l;
      }
    draw:
      IntType j = (v / m) * m; v -= j; c -= j;
      v <<= bits; c <<= bits; c += IntType(RandomDigit(r));
    }
  }

  /**
   * \relates UniformInteger
   * Print a UniformInteger.  Format is [\e min,\e max] unless the entropy is
   * zero, in which case it's \e val.
   **********************************************************************/
  template<typename IntType, int bits>
  std::ostream& operator<<(std::ostream& os,
                           const UniformInteger<IntType, bits>& u) {
    if (u.Entropy())
      os << "[" << u.Min() << "," << u.Max() << "]";
    else
      os << u.Min();
    return os;
  }

} // namespace RandomLib

#endif  // RANDOMLIB_UNIFORMINTEGER_HPP
