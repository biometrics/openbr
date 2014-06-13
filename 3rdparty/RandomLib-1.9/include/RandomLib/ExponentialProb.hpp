/**
 * \file ExponentialProb.hpp
 * \brief Header for ExponentialProb
 *
 * Return true with probabililty exp(-\e p).
 *
 * Copyright (c) Charles Karney (2006-2012) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_EXPONENTIALPROB_HPP)
#define RANDOMLIB_EXPONENTIALPROB_HPP 1

#include <vector>
#include <limits>

#if defined(_MSC_VER)
// Squelch warnings about constant conditional expressions
#  pragma warning (push)
#  pragma warning (disable: 4127)
#endif

namespace RandomLib {
  /**
   * \brief The exponential probability.
   *
   * Return true with probability exp(&minus;\e p).  Basic method taken from:\n
   * J. von Neumann,\n Various Techniques used in Connection with Random
   * Digits,\n J. Res. Nat. Bur. Stand., Appl. Math. Ser. 12, 36--38
   * (1951),\n reprinted in Collected Works, Vol. 5, 768--770 (Pergammon,
   * 1963).\n See also the references given for the ExactExponential class.
   *
   * Here the method is extended to be exact by generating sufficient bits in
   * the random numbers in the algorithm to allow the unambiguous comparisons
   * to be made.
   *
   * Here's one way of sampling from a normal distribution with zero mean and
   * unit variance in the interval [&minus;1,1] with reasonable accuracy:
   * \code
   #include <RandomLib/Random.hpp>
   #include <RandomLib/ExponentialProb.hpp>

   double Normal(RandomLib::Random& r) {
     double x;
     RandomLib::ExponentialProb e;
     do
        x = r.FloatW();
     while ( !e(r, - 0.5 * x * x) );
     return x;
   }
   \endcode
   * (Note that the ExactNormal class samples from the normal distribution
   * exactly.)
   *
   * This class uses a mutable private vector.  So a single ExponentialProb
   * object cannot safely be used by multiple threads.  In a multi-processing
   * environment, each thread should use a thread-specific ExponentialProb
   * object.
   **********************************************************************/
  class ExponentialProb {
  private:
    typedef unsigned word;
  public:

    ExponentialProb() : _v(std::vector<word>(alloc_incr)) {}
    /**
     * Return true with probability exp(&minus;\e p).  Returns false if \e p
     * &le; 0.  For in \e p (0,1], it requires about exp(\e p) random deviates.
     * For \e p large, it requires about exp(1)/(1 &minus; exp(&minus;1))
     * random deviates.
     *
     * @tparam RealType the real type of the argument.
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in] p the probability.
     * @return true with probability \e p.
     **********************************************************************/
    template<typename RealType, class Random>
    bool operator()(Random& r, RealType p) const;

  private:
    /**
     * Return true with probability exp(&minus;\e p) for \e p in [0,1].
     **********************************************************************/
    template<typename RealType, class Random>
    bool ExpFraction(Random& r, RealType p) const;
    /**
     * Holds as much of intermediate uniform deviates as needed.
     **********************************************************************/
    mutable std::vector<word> _v;
    /**
     * Increment on size of _v.
     **********************************************************************/
    static const unsigned alloc_incr = 16;
  };

  template<typename RealType, class Random>
  bool ExponentialProb::operator()(Random& r, RealType p) const {
    STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer,
                  "ExponentialProb(): invalid real type RealType");
    return p <= 0 ||            // True if p <=0
      // Ensure p - 1 < p.  Also deal with IsNaN(p)
      ( p < RealType(1)/std::numeric_limits<RealType>::epsilon() &&
        // exp(a+b) = exp(a) * exp(b)
        ExpFraction(r, p < RealType(1) ? p : RealType(1)) &&
        ( p <= RealType(1) || operator()(r, p - RealType(1)) ) );
  }

  template<typename RealType, class Random>
  bool ExponentialProb::ExpFraction(Random& r, RealType p) const {
    // Base of _v is 2^c.  Adjust so that word(p) doesn't lose precision.
    static const int c =        // The Intel compiler needs this to be static??
      std::numeric_limits<word>::digits <
      std::numeric_limits<RealType>::digits ?
      std::numeric_limits<word>::digits :
      std::numeric_limits<RealType>::digits;
    // m gives number of valid words in _v
    unsigned m = 0, l = unsigned(_v.size());
    if (p < RealType(1))
      while (true) {
        if (p <= RealType(0))
          return true;
        // p in (0, 1)
        if (l == m)
          _v.resize(l += alloc_incr);
        _v[m++] = r.template Integer<word, c>();
        p *= std::pow(RealType(2), c); // p in (0, 2^c)
        word w = word(p);              // w in [0, 2^c)
        if (_v[m - 1] > w)
          return true;
        else if (_v[m - 1] < w)
          break;
        else                    // _v[m - 1] == w
          p -= RealType(w);     // p in [0, 1)
      }
    // Here _v < p.  Now loop finding decreasing V.  Exit when first increasing
    // one is found.
    for (unsigned s = 0; ; s ^= 1) { // Parity of loop count
      for (unsigned j = 0; ; ++j) {
        if (j == m) {
          // Need more bits in the old V
          if (l == m)
            _v.resize(l += alloc_incr);
          _v[m++] = r.template Integer<word, c>();
        }
        word w = r.template Integer<word, c>();
        if (w > _v[j])
          return s != 0u;             // New V is bigger, so exit
        else if (w < _v[j]) {
          _v[j] = w;            // New V is smaller, update _v
          m = j + 1;            // adjusting its size
          break;                // and generate the next V
        }
        // Else w == _v[j] and we need to check the next c bits
      }
    }
  }

} // namespace RandomLib

#if defined(_MSC_VER)
#  pragma warning (pop)
#endif

#endif  // RANDOMLIB_EXPONENTIALPROB_HPP
