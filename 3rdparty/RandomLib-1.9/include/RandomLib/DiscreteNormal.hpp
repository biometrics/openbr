/**
 * \file DiscreteNormal.hpp
 * \brief Header for DiscreteNormal
 *
 * Sample exactly from the discrete normal distribution.
 *
 * Copyright (c) Charles Karney (2013) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_DISCRETENORMAL_HPP)
#define RANDOMLIB_DISCRETENORMAL_HPP 1

#include <vector>
#include <limits>

namespace RandomLib {
  /**
   * \brief The discrete normal distribution.
   *
   * Sample integers \e i with probability proportional to
   * \f[
   * \exp\biggl[-\frac12\biggl(\frac{i-\mu}{\sigma}\biggr)^2\biggr],
   * \f]
   * where &sigma; and &mu; are given as rationals (the ratio of two integers).
   * The sampling is exact (provided that the random generator is ideal).  For
   * example
   * \code
   #include <iostream>
   #include <RandomLib/Random.hpp>
   #include <RandomLib/DiscreteNormal.hpp>

   int main() {
     RandomLib::Random r;          // Create r
     r.Reseed();                   // and give it a unique seed
     int sigma_num = 7, sigma_den = 1, mu_num = 1, mu_den = 3;
     RandomLib::DiscreteNormal<int> d(sigma_num, sigma_den,
                                      mu_num, mu_den);
     for (int i = 0; i < 100; ++i)
       std::cout << d(r) << "\n";
   }
   \endcode
   * prints out 100 samples with &sigma; = 7 and &mu; = 1/3.
   *
   * The algorithm is much the same as for ExactNormal; for details see
   * - C. F. F. Karney, <i>Sampling exactly from the normal distribution</i>,
   *   http://arxiv.org/abs/1303.6257 (Mar. 2013).
   * .
   * That algorithm samples the integer part of the result \e k, samples \e x
   * in [0,1], and (unless rejected) returns <i>s</i>(\e k + \e x), where \e s
   * = &plusmn;1.  For the discrete case, we sample \e x in [0,1) such that
   * \f[
   *  s(k + x) = (i - \mu)/\sigma,
   * \f]
   * or
   * \f[
   *   x = s(i - \mu)/\sigma - k
   * \f]
   * The value of \e i which results in the smallest \e x &ge; 0 is
   * \f[
   *   i_0 = s\lceil k \sigma + s \mu\rceil
   * \f]
   * so sample
   * \f[
   *   i = i_0 + sj
   * \f]
   * where \e j is uniformly distributed in [0, &lceil;&sigma;&rceil;).  The
   * corresponding value of \e x is
   * \f[
   * \begin{aligned}
   *   x &= \bigl(si_0 - (k\sigma + s\mu)\bigr)/\sigma + j/\sigma\\
   *     &= x_0 + j/\sigma,\\
   *   x_0 &= \bigl(\lceil k \sigma + s \mu\rceil -
   *                      (k \sigma + s \mu)\bigr)/\sigma.
   * \end{aligned}
   * \f]
   * After \e x is sampled in this way, it should be rejected if \e x &ge; 1
   * (this is counted with the next larger value of \e k) or if \e x = 0, \e k
   * = 0, and \e s = &minus;1 (to avoid double counting the origin).  If \e x
   * is accepted (in Step 4 of the ExactNormal algorithm), then return \e i.
   *
   * When &sigma; and &mu; are given as rationals, all the arithmetic outlined
   * above can be carried out exactly.  The basic rejection techniques used by
   * ExactNormal are exact.  Thus the result of this discrete form of the
   * algorithm is also exact.
   *
   * RandomLib provides two classes to sample from this distribution:
   * - DiscreteNormal which is tuned for speed on a typical general purpose
   *   computer.  This assumes that random samples can be generated relatively
   *   quickly.
   * - DiscreteNormalAlt, which is a prototype for what might be needed on a
   *   small device used for cryptography which is using a hardware generator
   *   for obtaining truly random bits.  This assumption here is that the
   *   random bits are relatively expensive to obtain.
   * .

   * The basic algorithm is the same in the two cases.  The main advantages of
   * this method are:
   * - exact sampling (provided that the source of random numbers is ideal),
   * - no need to cut off the tails of the distribution,
   * - a short program involving simple integer operations only,
   * - no dependence on external libraries (except to generate random bits),
   * - no large tables of constants needed,
   * - minimal time to set up for a new &sigma; and &mu; (roughly comparable to
   *   the time it takes to generate one sample),
   * - only about 5&ndash;20 times slower than standard routines to sample from
   *   a normal distribution using plain double-precision arithmetic.
   * - DiscreteNormalAlt exhibits ideal scaling for the consumption of random
   *   bits, namely a constant + log<sub>2</sub>&sigma;, for large &sigma;,
   *   where the constant is about 31.
   * .
   * The possible drawbacks of this method are:
   * - &sigma; and &mu; are restricted to rational numbers with sufficiently
   *   small numerators and denominators to avoid overflow (this is unlikely to
   *   be a severe restriction especially if the template parameter IntType is
   *   set to <code>long long</code>),
   * - the running time is unbounded (but not in any practical sense),
   * - the memory consumption is unbounded (but not in any practical sense),
   * - the toll, about 30 bits, is considerably worse than that obtained using
   *   the Knuth-Yao algorithm, for which the toll is no more than 2 (but this
   *   requires a large table which is expensive to compute and requires a lot
   *   of memory to store).
   *
   * This class uses a mutable private vector.  So a single DiscreteNormal
   * object cannot safely be used by multiple threads.  In a multi-processing
   * environment, each thread should use a thread-specific DiscreteNormal
   * object.
   *
   * Some timing results for IntType = int, &mu; = 0, and 10<sup>8</sup>
   * samples (time = time per sample, including setup time, rv = mean number of
   * random variables per sample)
   * - &sigma; = 10, time = 219 ns, rv = 17.52
   * - &sigma; = 32, time = 223 ns, rv = 17.82
   * - &sigma; = 1000, time = 225 ns, rv = 17.95
   * - &sigma; = 160000, time = 226 ns, rv = 17.95
   *
   * @tparam IntType the integer type to use (default int).
   **********************************************************************/
  template<typename IntType = int>  class DiscreteNormal {
  public:
    /**
     * Constructor.
     *
     * @param[in] sigma_num the numerator of &sigma;.
     * @param[in] sigma_den the denominator of &sigma; (default 1).
     * @param[in] mu_num the numerator of &mu; (default 0).
     * @param[in] mu_den the denominator of &mu; (default 1).
     *
     * The constructor creates a DiscreteNormal objects for sampling with
     * specific values of &sigma; and &mu;.  This may throw an exception if the
     * parameters are such that overflow is possible.  Internally &sigma; and
     * &mu; are expressed with a common denominator, so it may be possible to
     * avoid overflow by picking the fractions of these quantities so that \e
     * sigma_den and \e mu_den have many common factors.
     **********************************************************************/
    DiscreteNormal(IntType sigma_num, IntType sigma_den = 1,
                   IntType mu_num = 0, IntType mu_den = 1);
    /**
     * Return a sample.
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @return discrete normal integer.
     **********************************************************************/
    template<class Random>
    IntType operator()(Random& r) const;
  private:
    /**
     * sigma = _sig / _d, mu = _imu + _mu / _d, _isig = floor(sigma)
     **********************************************************************/
    IntType _sig, _mu, _d, _isig, _imu;
    typedef unsigned short word;
    /**
     * Holds as much of intermediate uniform deviates as needed.
     **********************************************************************/
    mutable std::vector<word> _v;
    mutable unsigned _m, _l;
    /**
     * Increment on size of _v.
     **********************************************************************/
    static const unsigned alloc_incr = 16;

    // ceil(n/d) for d > 0
    static IntType iceil(IntType n, IntType d);
    // abs(n) needed because Visual Studio's std::abs has problems
    static IntType iabs(IntType n);
    static IntType gcd(IntType u, IntType v);

    // After x = LeadingDigit(p), p/_sig = (x + p'/_sig)/b where p and p' are
    // in [0, _sig) and b = 1 + max(word).
    word LeadingDigit(IntType& p) const;

    /**
     * Implement outcomes for choosing with prob (\e x + 2\e k) / (2\e k + 2);
     * return:
     * - 1 (succeed unconditionally) with prob (\e m &minus; 2) / \e m,
     * - 0 (succeed with probability x) with prob 1 / \e m,
     * - &minus;1 (fail unconditionally) with prob 1 / \e m.
     **********************************************************************/
    template<class Random> static int Choose(Random& r, int m);

    // Compute v' < v.  If true set v = v'.
    template<class Random> bool less_than(Random& r) const;

    // Compute v < (x + p/_sig)/base (updating v)
    template<class Random> bool less_than(Random& r, word x, IntType p) const;

    // true with prob (x + p/_sig)/base
    template<class Random> bool bernoulli(Random& r, word x, IntType p) const;

    /**
     * Return true with probability exp(&minus;1/2).
     **********************************************************************/
    template<class Random> bool ExpProbH(Random& r) const;

    /**
     * Return true with probability exp(&minus;<i>n</i>/2).
     **********************************************************************/
    template<class Random> bool ExpProb(Random& r, int n) const;

    /**
     * Return \e n with probability exp(&minus;<i>n</i>/2)
     * (1&minus;exp(&minus;1/2)).
     **********************************************************************/
    template<class Random> int ExpProbN(Random& r) const;

    /**
     * Algorithm B: true with prob exp(-x * (2*k + x) / (2*k + 2)) where
     * x = (x0 + xn / _sig)/b.
     **********************************************************************/
    template<class Random>
    bool B(Random& r, int k, word x0, IntType xn) const;
  };

  template<typename IntType> DiscreteNormal<IntType>::DiscreteNormal
  (IntType sigma_num, IntType sigma_den,
   IntType mu_num, IntType mu_den)
    : _v(std::vector<word>(alloc_incr)), _m(0), _l(alloc_incr) {
    STATIC_ASSERT(std::numeric_limits<IntType>::is_integer,
                  "DiscreteNormal: invalid integer type IntType");
    STATIC_ASSERT(std::numeric_limits<IntType>::is_signed,
                  "DiscreteNormal: IntType must be a signed type");
    STATIC_ASSERT(!std::numeric_limits<word>::is_signed,
                  "DiscreteNormal: word must be an unsigned type");
    STATIC_ASSERT(std::numeric_limits<IntType>::digits + 1 >=
                  std::numeric_limits<word>::digits,
                  "DiscreteNormal: IntType must be at least as wide as word");
    if (!( sigma_num > 0 && sigma_den > 0 && mu_den > 0 ))
      throw RandomErr("DiscreteNormal: need sigma > 0");
    _imu = mu_num / mu_den;
    if (_imu == (std::numeric_limits<IntType>::min)())
      throw RandomErr("DiscreteNormal: abs(mu) too large");
    mu_num -= _imu * mu_den;
    IntType l;
    l = gcd(sigma_num, sigma_den); sigma_num /= l; sigma_den /= l;
    l = gcd(mu_num, mu_den); mu_num /= l; mu_den /= l;
    _isig = iceil(sigma_num, sigma_den);
    l = gcd(sigma_den, mu_den);
    _sig = sigma_num * (mu_den / l);
    _mu = mu_num * (sigma_den / l);
    _d  = sigma_den * (mu_den / l);
    // The rest of the constructor tests for possible overflow
    // Check for overflow in computing member variables
    IntType maxint = (std::numeric_limits<IntType>::max)();
    if (!( mu_den / l <= maxint / sigma_num &&
           mu_num <= maxint / (sigma_den / l) &&
           mu_den / l <= maxint / sigma_den ))
      throw RandomErr("DiscreteNormal: sigma or mu overflow");
    // The probability that k =  kmax is about 10^-543.
    int kmax = 50;
    // Check that max plausible result fits in an IntType, i.e.,
    // _isig * (kmax + 1) + abs(_imu) does not lead to overflow.
    if (!( kmax + 1 <=  maxint / _isig &&
           _isig * (kmax + 1) <= maxint - iabs(_imu) ))
      throw RandomErr("DiscreteNormal: possible overflow a");
    // Check xn0 = _sig * k + s * _mu;
    if (!( kmax <= maxint / _sig &&
           _sig * kmax <= maxint - iabs(_mu) ))
      throw RandomErr("DiscreteNormal: possible overflow b");
    // Check for overflow in LeadingDigit
    // p << bits, p = _sig - 1, bits = 8
    if (!( _sig <= (maxint >> 8) ))
      throw RandomErr("DiscreteNormal: overflow in LeadingDigit");
  }

  template<typename IntType> template<class Random>
  IntType DiscreteNormal<IntType>::operator()(Random& r) const {
    for (;;) {
      int k = ExpProbN(r);
      if (!ExpProb(r, k * (k - 1))) continue;
      IntType
        s = r.Boolean() ? -1 : 1,
        xn = _sig * IntType(k) + s * _mu,
        i = iceil(xn, _d) + r.template Integer<IntType>(_isig);
      xn = i * _d - xn;
      if (xn >= _sig || (k == 0 && s < 0 && xn <= 0)) continue;
      if (xn > 0) {
        word x0 = LeadingDigit(xn); // Find first digit in expansion in words
        int h = k + 1; while (h-- && B(r, k, x0, xn));
        if (!(h < 0)) continue;
      }
      return s * i + _imu;
    }
  }

  template<typename IntType>
  IntType DiscreteNormal<IntType>::iceil(IntType n, IntType d)
  { IntType k = n / d; return k + (k * d < n ? 1 : 0); }

  template<typename IntType> IntType DiscreteNormal<IntType>::iabs(IntType n)
  { return n < 0 ? -n : n; }

  template<typename IntType>
  IntType DiscreteNormal<IntType>::gcd(IntType u, IntType v) {
    // Knuth, TAOCP, vol 2, 4.5.2, Algorithm A
    u = iabs(u); v = iabs(v);
    while (v > 0) { IntType r = u % v; u = v; v = r; }
    return u;
  }

  template<typename IntType> typename DiscreteNormal<IntType>::word
  DiscreteNormal<IntType>::LeadingDigit(IntType& p) const {
    static const unsigned bits = 8;
    static const unsigned num = std::numeric_limits<word>::digits / bits;
    STATIC_ASSERT(bits * num == std::numeric_limits<word>::digits,
          "Number of digits in word must be multiple of 8");
    word s = 0;
    for (unsigned c = num; c--;) {
      p <<= bits; s <<= bits;
      word d = word(p / _sig);
      s += d;
      p -= IntType(d) * _sig;
    }
    return s;
  }

  template<typename IntType> template<class Random>
  int DiscreteNormal<IntType>::Choose(Random& r, int m) {
    int k = r.template Integer<int>(m);
    return k == 0 ? 0 : (k == 1 ? -1 : 1);
  }

  template<typename IntType> template<class Random>
  bool DiscreteNormal<IntType>::less_than(Random& r) const {
    for (unsigned j = 0; ; ++j) {
      if (j == _m) {
        // Need more bits in the old V
        if (_l == _m) _v.resize(_l += alloc_incr);
        _v[_m++] = r.template Integer<word>();
      }
      word w = r.template Integer<word>();
      if (w > _v[j])
        return false;           // New V is bigger, so exit
      else if (w < _v[j]) {
        _v[j] = w;              // New V is smaller, update _v
        _m = j + 1;             // adjusting its size
        return true;            // and generate the next V
      }
      // Else w == _v[j] and we need to check the next word
    }
  }

  template<typename IntType> template<class Random>
  bool DiscreteNormal<IntType>::less_than(Random& r, word x, IntType p) const {
    if (_m == 0) _v[_m++] = r.template Integer<word>();
    if (_v[0] != x) return _v[0] < x;
    for (unsigned j = 1; ; ++j) {
      if (p == 0) return false;
      if (j == _m) {
        if (_l == _m) _v.resize(_l += alloc_incr);
        _v[_m++] = r.template Integer<word>();
      }
      x = LeadingDigit(p);
      if (_v[j] != x) return _v[j] < x;
    }
  }

  template<typename IntType> template<class Random>
  bool DiscreteNormal<IntType>::bernoulli(Random& r, word x, IntType p) const {
    word w = r.template Integer<word>();
    if (w != x) return w < x;
    for (;;) {
      if (p == 0) return false;
      x = LeadingDigit(p);
      w = r.template Integer<word>();
      if (w != x) return w < x;
    }
  }

  template<typename IntType> template<class Random>
  bool DiscreteNormal<IntType>::ExpProbH(Random& r) const {
    static const word half = word(1) << (std::numeric_limits<word>::digits - 1);
    _m = 0;
    if ((_v[_m++] = r.template Integer<word>()) & half) return true;
    // Here _v < 1/2.  Now loop finding decreasing V.  Exit when first
    // increasing one is found.
    for (unsigned s = 0; ; s ^= 1) { // Parity of loop count
      if (!less_than(r)) return s != 0u;
    }
  }

  template<typename IntType> template<class Random>
  bool DiscreteNormal<IntType>::ExpProb(Random& r, int n) const {
    while (n-- > 0) { if (!ExpProbH(r)) return false; }
    return true;
  }

  template<typename IntType> template<class Random>
  int DiscreteNormal<IntType>::ExpProbN(Random& r) const {
    int n = 0;
    while (ExpProbH(r)) ++n;
    return n;
  }

  template<typename IntType> template<class Random>
  bool DiscreteNormal<IntType>::B(Random& r, int k, word x0, IntType xn)
    const {
    int n = 0, h = 2 * k + 2, f;
    _m = 0;
    for (;; ++n) {
      if ( ((f = k ? 0 : Choose(r, h)) < 0) ||
           !(n ? less_than(r) : less_than(r, x0, xn)) ||
           ((f = k ? Choose(r, h) : f) < 0) ||
           (f == 0 && !bernoulli(r, x0, xn)) ) break;
    }
    return (n % 2) == 0;
  }

} // namespace RandomLib

#endif  // RANDOMLIB_DISCRETENORMAL_HPP
