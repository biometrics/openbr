/**
 * \file RandomNumber.hpp
 * \brief Header for RandomNumber
 *
 * Infinite precision random numbers.
 *
 * Copyright (c) Charles Karney (2006-2013) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_RANDOMNUMBER_HPP)
#define RANDOMLIB_RANDOMNUMBER_HPP 1

#include <vector>
#include <iomanip>
#include <limits>
#include <cmath>                // for std::pow
#include <RandomLib/UniformInteger.hpp>

namespace RandomLib {
  /**
   * \brief Infinite precision random numbers.
   *
   * Implement infinite precision random numbers.  Integer part is non-random.
   * Fraction part consists of any some number of digits in base
   * 2<sup><i>b</i></sup>.  If \e m digits have been generated then the
   * fraction is uniformly distributed in the open interval
   * &sum;<sub><i>k</i>=1</sub><sup><i>m</i></sup>
   * <i>f</i><sub><i>k</i>&minus;1</sub>/2<sup><i>kb</i></sup> +
   * (0,1)/2<sup><i>mb</i></sup>.  When a RandomNumber is first constructed the
   * integer part is zero and \e m = 0, and the number represents (0,1).  A
   * RandomNumber is able to represent all numbers in the symmetric open
   * interval (&minus;2<sup>31</sup>, 2<sup>31</sup>).  In this implementation,
   * \e b must one of 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, or 32.  (This
   * restriction allows printing in hexadecimal and can easily be relaxed.
   * There's also no essential reason why the base should be a power of 2.)
   *
   * @tparam bits the number of bits in each digit.
   **********************************************************************/
  template<int bits = 1> class RandomNumber {
  public:
    /**
     * Constructor sets number to a random number uniformly distributed in
     * (0,1).
     **********************************************************************/
    RandomNumber() throw() : _n(0), _s(1) {}
    /**
     * Swap with another RandomNumber.  This is a fast way of doing an
     * assignment.
     *
     * @param[in,out] t the RandomNumber to swap with.
     **********************************************************************/
    void swap(RandomNumber& t) throw() {
      if (this != &t) {
        std::swap(_n, t._n);
        std::swap(_s, t._s);
        _f.swap(t._f);
      }
    }
    /**
     * Return to initial state, uniformly distributed in (0,1).
     **********************************************************************/
    void Init() throw() {
      STATIC_ASSERT(bits > 0 && bits <= w && (bits < 4 || bits % 4 == 0),
                    "RandomNumber: unsupported value for bits");
      _n = 0;
      _s = 1;
      _f.clear();
    }
    /**
     * @return the sign of the RandomNumber (&plusmn; 1).
     **********************************************************************/
    int Sign() const throw() { return _s; }
    /**
     * Change the sign of the RandomNumber.
     **********************************************************************/
    void Negate() throw() { _s *= -1; }
    /**
     * @return the floor of the RandomNumber.
     **********************************************************************/
    int Floor() const throw() { return _s > 0 ? int(_n) : -1 - int(_n); }
    /**
     * @return the ceiling of the RandomNumber.
     **********************************************************************/
    int Ceiling() const throw() { return _s > 0 ? 1 + int(_n) : - int(_n); }
    /**
     * @return the unsigned integer component of the RandomNumber.
     **********************************************************************/
    unsigned UInteger() const throw() { return _n; }
    /**
     * Add integer \e k to the RandomNumber.
     *
     * @param[in] k the integer to add.
     **********************************************************************/
    void AddInteger(int k) throw() {
      k += Floor();             // The new floor
      int ns = k < 0 ? -1 : 1;  // The new sign
      if (ns != _s)             // If sign changes, set f = 1 - f
        for (size_t k = 0; k < Size(); ++k)
          _f[k] = ~_f[k] & mask;
      _n = ns > 0 ? k : -(k + 1);
    }
    /**
     * Compare with another RandomNumber, *this &lt; \e t
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in,out] t a RandomNumber to compare.
     * @return true if *this &lt; \e t.
     **********************************************************************/
    template<class Random> bool LessThan(Random& r, RandomNumber& t) {
      if (this == &t) return false; // same object
      if (_s != t._s) return _s < t._s;
      if (_n != t._n) return (_s < 0) ^ (_n < t._n);
      for (unsigned k = 0; ; ++k) {
        // Impose an order on the evaluation of the digits.
        const unsigned x = Digit(r,k);
        const unsigned y = t.Digit(r,k);
        if (x != y) return (_s < 0) ^ (x < y);
        // Two distinct numbers are never equal
      }
    }
    /**
     * Compare RandomNumber with two others, *this &gt; max(\e u, \e v)
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in,out] u first RandomNumber to compare.
     * @param[in,out] v second RandomNumber to compare.
     * @return true if *this &gt; max(\e u, \e v).
     **********************************************************************/
    template<class Random> bool GreaterPair(Random& r,
                                            RandomNumber& u, RandomNumber& v) {
      // cmps is set to false as soon as u <= *this, and likewise for cmpt.
      bool cmpu = this != &u, cmpv = this != &v && &u != &v;
      if (!(cmpu || cmpv)) return true;
      // Check signs first
      if (cmpu) {
        if (u._s > _s) return false; // u > *this
        if (u._s < _s) cmpu = false;
      }
      if (cmpv) {
        if (v._s > _s) return false; // v > *this
        if (v._s < _s) cmpv = false;
      }
      if (!(cmpu || cmpv)) return true; // u <= *this && v <= *this
      // Check integer parts
      if (cmpu) {
        if ((_s < 0) ^ (u._n > _n)) return false; // u > *this
        if ((_s < 0) ^ (u._n < _n)) cmpu = false;
      }
      if (cmpv) {
        if ((_s < 0) ^ (v._n > _n)) return false; // v > *this
        if ((_s < 0) ^ (v._n < _n)) cmpv = false;
      }
      if (!(cmpu || cmpv)) return true; // u <= *this && v <= *this
      // Check fractions
      for (unsigned k = 0; ; ++k) {
        // Impose an order on the evaluation of the digits.  Note that this is
        // asymmetric on interchange of u and v; since u is tested first, more
        // digits of u are generated than v (on average).
        const unsigned x = Digit(r,k);
        if (cmpu) {
          const unsigned y = u.Digit(r,k);
          if ((_s < 0) ^ (y > x)) return false; // u > *this
          if ((_s < 0) ^ (y < x)) cmpu = false;
        }
        if (cmpv) {
          const unsigned y = v.Digit(r,k);
          if ((_s < 0) ^ (y > x)) return false; // v > *this
          if ((_s < 0) ^ (y < x)) cmpv = false;
        }
        if (!(cmpu || cmpv)) return true; // u <= *this && v <= *this
      }
    }
    /**
     * Compare with a fraction, *this &lt; <i>p</i>/<i>q</i>
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in] p the numerator of the fraction.
     * @param[in] q the denominator of the fraction (require \e q &gt; 0).
     * @return true if *this &lt; <i>p</i>/<i>q</i>.
     **********************************************************************/
    template<class Random, typename IntType>
    bool LessThan(Random& r, IntType p, IntType q) {
      for (int k = 0;; ++k) {
        if (p <= 0) return false;
        if (p >= q) return true;
        // Here p is in [1,q-1].  Need to avoid overflow in computation of
        // (q-1)<<bits and (2^bits-1)*q
        p = (p << bits) - Digit(r,k) * q;
      }
    }
    /**
     * Compare with a paritally sampled fraction
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in] p0 the starting point for the numerator.
     * @param[in] c the stride for the fraction (require \e c &gt; 0).
     * @param[in] q the denominator of the fraction (require \e q &gt; 0).
     * @param[in,out] j the increment for the numerator.
     * @return true if *this &lt; (<i>p</i><sub>0</sub> + <i>cj</i>)/<i>q</i>.
     **********************************************************************/
    template<class Random, typename IntType>
    bool LessThan(Random& r, IntType p0, IntType c, IntType q,
                  UniformInteger<IntType, bits>& j) {
      for (int k = 0;; ++k) {
        if (j.   LessThanEqual(r,   - p0, c)) return false;
        if (j.GreaterThanEqual(r, q - p0, c)) return true;
        p0 = (p0 << bits) - IntType(Digit(r,k)) * q;
        c <<= bits;
      }
    }

    /**
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in] k the index of a digit of the fraction
     * @return digit number \e k, generating it if necessary.
     **********************************************************************/
    template<class Random> unsigned Digit(Random& r, unsigned k) {
      ExpandTo(r, k + 1);
      return _f[k];
    }
    /**
     * Add one digit to the fraction.
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     **********************************************************************/
    template<class Random> void AddDigit(Random& r)
    { _f.push_back(RandomDigit(r)); }
    /**
     * @param[in] k the index of a digit of the fraction
     * @return a const reference to digit number \e k, without generating new
     *   digits.
     * @exception std::out_of_range if the digit hasn't been generated.
     **********************************************************************/
    const unsigned& RawDigit(unsigned k) const throw()
    { return (const unsigned&)(_f.at(k)); }
    /**
     * @param[in] k the index of a digit of the fraction
     * @return a non-const reference to digit number \e k, without generating
     *   new digits.
     * @exception std::out_of_range if the digit hasn't been generated.
     **********************************************************************/
    unsigned& RawDigit(unsigned k) throw()
    { return (unsigned&)(_f.at(k)); }
    /**
     * Return to initial state, uniformly distributed in \e n + (0,1).  This is
     * similar to Init but also returns the memory used by the object to the
     * system.  Normally Init should be used.
     **********************************************************************/
    void Clear() {
      std::vector<unsigned> z(0);
      _n = 0;
      _s = 1;
      _f.swap(z);
    }
    /**
     * @return the number of digits in fraction
     **********************************************************************/
    unsigned Size() const throw() { return unsigned(_f.size()); }
    /**
     * Return the fraction part of the RandomNumber as a floating point number
     * of type RealType rounded to the nearest multiple of
     * 1/2<sup><i>p</i></sup>, where \e p =
     * std::numeric_limits<RealType>::digits, and, if necessary, creating
     * additional digits of the number.
     *
     * @tparam RealType the floating point type to convert to.
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator for generating the necessary digits.
     * @return the fraction of the RandomNumber rounded to a RealType.
     **********************************************************************/
    template<typename RealType, typename Random> RealType Fraction(Random& r) {
      STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer,
                    "RandomNumber::Fraction: invalid real type RealType");
      const int d = std::numeric_limits<RealType>::digits;
      const int k = (d + bits - 1)/bits;
      const int kg = (d + bits)/bits; // For guard bit
      RealType y = 0;
      if (Digit(r, kg - 1) & (1U << (kg * bits - d - 1)))
        // if guard bit is set, round up.
        y += std::pow(RealType(2), -d);
      const RealType fact = std::pow(RealType(2), -bits);
      RealType mult = RealType(1);
      for (int i = 0; i < k; ++i) {
        mult *= fact;
        y += mult * RealType(i < k - 1 ? RawDigit(i) :
                             RawDigit(i) & (~0U << (k * bits - d)));
      }
      return y;
    }
    /**
     * Return the value of the RandomNumber rounded to nearest floating point
     * number of type RealType and, if necessary, creating additional digits of
     * the number.
     *
     * @tparam RealType the floating point type to convert to.
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator for generating the necessary digits.
     * @return the value of the RandomNumber rounded to a RealType.
     **********************************************************************/
    template<typename RealType, class Random> RealType Value(Random& r) {
      // Ignore the possibility of overflow here (OK because int doesn't
      // currently overflow any real type).  Assume the real type supports
      // denormalized numbers.  Need to treat rounding explicitly since the
      // missing digits always imply rounding up.
      STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer,
                    "RandomNumber::Value: invalid real type RealType");
      const int digits = std::numeric_limits<RealType>::digits,
        min_exp = std::numeric_limits<RealType>::min_exponent;
      RealType y;
      int lead;               // Position of leading bit (0.5 = position 0)
      if (_n) lead = highest_bit_idx(_n);
      else {
        int i = 0;
        while ( Digit(r, i) == 0 && i < (-min_exp)/bits ) ++i;
        lead = highest_bit_idx(RawDigit(i)) - (i + 1) * bits;
        // To handle denormalized numbers set lead = max(lead, min_exp)
        lead = lead > min_exp ? lead : min_exp;
      }
      int trail = lead - digits; // Position of guard bit (0.5 = position 0)
      if (trail > 0) {
        y = RealType(_n & (~0U << trail));
        if (_n & (1U << (trail - 1)))
          y += std::pow(RealType(2), trail);
      } else {
        y = RealType(_n);
        int k = (-trail)/bits;  // Byte with guard bit
        if (Digit(r, k) & (1U << ((k + 1) * bits + trail - 1)))
          // If guard bit is set, round bit (some subsequent bit will be 1).
          y += std::pow(RealType(2), trail);
        // Byte with trailing bit (can be negative)
        k = (-trail - 1 + bits)/bits - 1;
        const RealType fact = std::pow(RealType(2), -bits);
        RealType mult = RealType(1);
        for (int i = 0; i <= k; ++i) {
          mult *= fact;
          y += mult *
            RealType(i < k ? RawDigit(i) :
                     RawDigit(i) & (~0U << ((k + 1) * bits + trail)));
        }
      }
      if (_s < 0) y *= -1;
      return y;
    }
    /**
     * Return the range of possible values for the RandomNumber as pair of
     * doubles.  This doesn't create any additional digits of the result and
     * doesn't try to control roundoff.
     *
     * @return a pair denoting the range with first being the lower limit and
     *   second being the upper limit.
     **********************************************************************/
    std::pair<double, double> Range() const throw() {
      double y = _n;
      const double fact = std::pow(double(2), -bits);
      double mult = double(1);
      for (unsigned i = 0; i < Size(); ++i) {
        mult *= fact;
        y += mult * RawDigit(i);
      }
      return std::pair<double, double>(_s > 0 ? y : -(y + mult),
                                       _s > 0 ? (y + mult) : -y);
    }
    /**
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @return a random digit in [0, 2<sup><i>bits</i></sup>).
     **********************************************************************/
    template<class Random> static unsigned RandomDigit(Random& r) throw()
    { return unsigned(r.template Integer<bits>()); }

  private:
    /**
     * The integer part
     **********************************************************************/
    unsigned _n;
    /**
     * The sign
     **********************************************************************/
    int _s;
    /**
     * The fraction part
     **********************************************************************/
    std::vector<unsigned> _f;
    /**
     * Fill RandomNumber to \e k digits.
     **********************************************************************/
    template<class Random> void ExpandTo(Random& r, size_t k) {
      size_t l = _f.size();
      if (k <= l)
        return;
      _f.resize(k);
      for (size_t i = l; i < k; ++i)
        _f[i] = RandomDigit(r);
    }
    /**
     * Return index [0..32] of highest bit set.  Return 0 if x = 0, 32 is if x
     * = ~0.  (From Algorithms for programmers by Joerg Arndt.)
     **********************************************************************/
    static int highest_bit_idx(unsigned x) throw() {
      if (x == 0) return 0;
      int r = 1;
      if (x & 0xffff0000U) { x >>= 16; r += 16; }
      if (x & 0x0000ff00U) { x >>=  8; r +=  8; }
      if (x & 0x000000f0U) { x >>=  4; r +=  4; }
      if (x & 0x0000000cU) { x >>=  2; r +=  2; }
      if (x & 0x00000002U) {           r +=  1; }
      return r;
    }
    /**
     * The number of bits in unsigned.
     **********************************************************************/
    static const int w = std::numeric_limits<unsigned>::digits;
  public:
    /**
     * A mask for the digits.
     **********************************************************************/
    static const unsigned mask =
      bits == w ? ~0U : ~(~0U << (bits < w ? bits : 0));
  };

  /**
   * \relates RandomNumber
   * Print a RandomNumber.  Format is n.dddd... where the base for printing is
   * 2<sup>max(4,<i>b</i>)</sup>.  The ... represents an infinite sequence of
   * ungenerated random digits (uniformly distributed).  Thus with \e b = 1,
   * 0.0... = (0,1/2), 0.00... = (0,1/4), 0.11... = (3/4,1), etc.
   **********************************************************************/
  template<int bits>
  std::ostream& operator<<(std::ostream& os, const RandomNumber<bits>& n) {
    const std::ios::fmtflags oldflags = os.flags();
    RandomNumber<bits> t = n;
    os << (t.Sign() > 0 ? "+" : "-");
    unsigned i = t.UInteger();
    os << std::hex << std::setfill('0');
    if (i == 0)
      os << "0";
    else {
      bool first = true;
      const int w = std::numeric_limits<unsigned>::digits;
      const unsigned mask = RandomNumber<bits>::mask;
      for (int s = ((w + bits - 1)/bits) * bits - bits; s >= 0; s -= bits) {
        unsigned d = mask & (i >> s);
        if (d || !first) {
          if (first) {
            os << d;
            first = false;
          }
          else
            os << std::setw((bits+3)/4) << d;
        }
      }
    }
    os << ".";
    unsigned s = t.Size();
    for (unsigned i = 0; i < s; ++i)
      os << std::setw((bits+3)/4) << t.RawDigit(i);
    os << "..." << std::setfill(' ');
    os.flags(oldflags);
    return os;
  }

} // namespace RandomLib

#endif  // RANDOMLIB_RANDOMNUMBER_HPP
