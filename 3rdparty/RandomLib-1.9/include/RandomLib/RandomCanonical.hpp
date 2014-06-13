/**
 * \file RandomCanonical.hpp
 * \brief Header for RandomCanonical.
 *
 * Use the random bits from Generator to produce random integers of various
 * sizes, random reals with various precisions, a random probability, etc.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_RANDOMCANONICAL_HPP)
#define RANDOMLIB_RANDOMCANONICAL_HPP 1

#include <bitset>
#include <RandomLib/RandomPower2.hpp>
#include <RandomLib/RandomEngine.hpp>

#if defined(_MSC_VER)
// Squelch warnings about constant conditional expressions and casts truncating
// constants
#  pragma warning (push)
#  pragma warning (disable: 4127 4310)
#endif

namespace RandomLib {
  /**
   * \brief Generate random integers, reals, and booleans.
   *
   * Use the random bits from Generator to produce random integers of various
   * sizes, random reals with various precisions, a random probability, etc.
   * RandomCanonical assumes that Generator produces random results as 32-bit
   * quantities (of type uint32_t) via Generator::Ran32(), 64-bit quantities
   * (of type uint64_t) via Generator::Ran64(), and in "natural" units of
   * Generator::width bits (of type Generator::result_type) via
   * Generator::Ran().
   *
   * For the most part this class uses Ran() when needing \e width or fewer
   * bits, otherwise it uses Ran64().  However, when \e width = 64, the
   * resulting code is RandomCanonical::Unsigned(\e n) is inefficient because
   * of the 64-bit arithmetic.  For this reason RandomCanonical::Unsigned(\e n)
   * uses Ran32() if less than 32 bits are required (even though this results
   * in more numbers being produced by the Generator).
   *
   * This class has been tested with the 32-bit and 64-bit versions of MT19937
   * and SFMT19937.  Other random number generators could be used provided that
   * they provide a whole number of random bits so that Ran() is uniformly
   * distributed in [0,2<sup><i>w</i></sup>).  Probably some modifications
   * would be needed if \e w is not 32 or 64.
   *
   * @tparam Generator the type of the underlying generator.
   **********************************************************************/
  template<class Generator>
  class RandomCanonical : public Generator {
  public:
    /**
     * The type of operator()().
     **********************************************************************/
    typedef typename Generator::result_type result_type;
    /**
     * The type of elements of Seed().
     **********************************************************************/
    typedef typename RandomSeed::seed_type seed_type;
    enum {
      /**
       * The number of random bits in result_type.
       **********************************************************************/
      width = Generator::width
    };

    /**
     * \name Constructors which set the seed
     **********************************************************************/
    ///@{
    /**
     * Initialize from a vector.
     *
     * @tparam IntType the integral type of the elements of the vector.
     * @param[in] v the vector of elements.
     **********************************************************************/
    template<typename IntType>
    explicit RandomCanonical(const std::vector<IntType>& v) : Generator(v) {}
    /**
     * Initialize from a pair of iterator setting seed to [\e a, \e b)
     *
     * @tparam InputIterator the type of the iterator.
     * @param[in] a the beginning iterator.
     * @param[in] b the ending iterator.
     **********************************************************************/
    template<typename InputIterator>
    RandomCanonical(InputIterator a, InputIterator b) : Generator(a, b) {}
    /**
     * Initialize with seed [\e n]
     *
     * @param[in] n the new seed to use.
     **********************************************************************/
    explicit RandomCanonical(seed_type n);
    /**
     * Initialize with seed [].  This can be followed by a call to Reseed() to
     * select a unique seed.
     **********************************************************************/
    RandomCanonical() : Generator() {}
    /**
     * Initialize from a string.  See RandomCanonical::StringToVector
     *
     * @param[in] s the string to be decoded into a seed.
     **********************************************************************/
    explicit RandomCanonical(const std::string& s) : Generator(s) {}
    ///@}

    /**
     * \name Member functions returning integers
     **********************************************************************/
    ///@{
    /**
     * Return a raw result in [0, 2<sup><i>w</i></sup>) from the
     * underlying Generator.
     *
     * @return a <i>w</i>-bit random number.
     **********************************************************************/
    result_type operator()() throw() { return Generator::Ran(); }

    /**
     * A random integer in [0, \e n).  This allows a RandomCanonical object to
     * be passed to those standard template library routines that require
     * random numbers.  E.g.,
     * \code
       RandomCanonical r;
       int a[] = {0, 1, 2, 3, 4};
       std::random_shuffle(a, a+5, r);
     \endcode
     *
     * @param[in] n the upper end of the interval.  The upper end of the
     *   interval is open, so \e n is never returned.
     * @return the random integer in [0, \e n).
     **********************************************************************/
    result_type operator()(result_type n) throw()
    { return Integer<result_type>(n); }

    // Integer results (binary range)

    /**
     * A random integer of type IntType in [0, 2<sup><i>b</i></sup>).
     *
     * @tparam IntType the integer type of the returned random numbers.
     * @tparam bits how many random bits to return.
     * @return the random result.
     **********************************************************************/
    template<typename IntType, int bits> IntType Integer() throw() {
      // A random integer of type IntType in [0, 2^bits)
      STATIC_ASSERT(std::numeric_limits<IntType>::is_integer &&
                    std::numeric_limits<IntType>::radix == 2,
                    "Integer<T,b>(): bad integer type IntType");
      // Check that we have enough digits in Ran64
      STATIC_ASSERT(bits > 0 && bits <= std::numeric_limits<IntType>::digits &&
                    bits <= 64, "Integer<T,b>(): invalid value for bits");
      // Prefer masking to shifting so that we don't have to worry about sign
      // extension (a non-issue, because Ran/64 are unsigned?).
      return bits <= width ?
        IntType(Generator::Ran() & Generator::mask
                >> (bits <= width ? width - bits : 0)) :
        IntType(Generator::Ran64() & u64::mask >> (64 - bits));
    }

    /**
     * A random integer in [0, 2<sup><i>b</i></sup>).
     *
     * @tparam bits how many random bits to return.
     * @return the random result.
     **********************************************************************/
    template<int bits>
    result_type Integer() throw() { return Integer<result_type, bits>(); }

    /**
     * A random integer of type IntType in
     * [std::numeric_limits<IntType>::min(), std::numeric_limits::max()].
     *
     * @tparam IntType the integer type of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename IntType> IntType Integer() throw();

    /**
     * A random result_type in [0, std::numeric_limits<result_type>::max()].
     *
     * @return the random result.
     **********************************************************************/
    result_type Integer() throw()
    { return Integer<result_type>(); }

    // Integer results (finite range)

    /**
     * A random integer of type IntType in [0, \e n). \e Excludes \e n.  If \e
     * n == 0, treat as std::numeric_limits::max() + 1.  If \e n < 0, return 0.
     * Compare RandomCanonical::Integer<int>(0) which returns a result in
     * [0,2<sup>31</sup>) with RandomCanonical::Integer<int>() which returns a
     * result in [&minus;2<sup>31</sup>,2<sup>31</sup>).
     *
     * @tparam IntType the integer type of the returned random numbers.
     * @param[in] n the upper end of the semi-open interval.
     * @return the random result in [0, \e n).
     **********************************************************************/
    template<typename IntType> IntType Integer(IntType n) throw();
    /**
     * A random integer of type IntType in Closed interval [0, \e n].  \e
     * Includes \e n.  If \e n < 0, return 0.
     *
     * @tparam IntType the integer type of the returned random numbers.
     * @param[in] n the upper end of the closed interval.
     * @return the random result in [0, \e n].
     **********************************************************************/
    template<typename IntType> IntType IntegerC(IntType n) throw();
    /**
     * A random integer of type IntType in Closed interval [\e m, \e n].  \e
     * Includes both endpoints.  If \e n < \e m, return \e m.
     *
     * @tparam IntType the integer type of the returned random numbers.
     * @param[in] m the lower end of the closed interval.
     * @param[in] n the upper end of the closed interval.
     * @return the random result in [\e m, \e n].
     **********************************************************************/
    template<typename IntType> IntType IntegerC(IntType m, IntType n) throw();
    ///@}

    /**
     * \name Member functions returning real fixed-point numbers
     **********************************************************************/
    ///@{
    /**
     * In the description of the functions FixedX returning \ref fixed
     * "fixed-point" numbers, \e u is a random real number uniformly
     * distributed in (0, 1), \e p is the precision, and \e h =
     * 1/2<sup><i>p</i></sup>.  Each of the functions come in three variants,
     * e.g.,
     *   - RandomCanonical::Fixed<RealType,p>() --- return \ref fixed
     *     "fixed-point" real of type RealType, precision \e p;
     *   - RandomCanonical::Fixed<RealType>() --- as above with \e p =
     *     std::numeric_limits<RealType>::digits;
     *   - RandomCanonical::Fixed() --- as above with RealType = double.
     *
     * See the \ref reals "summary" for a comparison of the functions.
     *
     * Return \e i \e h with \e i in [0,2<sup><i>p</i></sup>) by rounding \e u
     * down to the previous \ref fixed "fixed" real.  Result is in default
     * interval [0,1).
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec> RealType Fixed() throw() {
      // RandomCanonical reals in [0, 1).  Results are of the form i/2^prec for
      // integer i in [0,2^prec).
      STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer &&
                    std::numeric_limits<RealType>::radix == 2,
                    "Fixed(): bad real type RealType");
      STATIC_ASSERT(prec > 0 && prec <= std::numeric_limits<RealType>::digits,
                    "Fixed(): invalid precision");
      RealType x = 0;           // Accumulator
      int s = 0;                // How many bits so far
      // Let n be the loop count.  Typically prec = 24, n = 1 for float; prec =
      // 53, n = 2 for double; prec = 64, n = 2 for long double.  For Sun
      // Sparc's, we have prec = 113, n = 4 for long double.  For Windows, long
      // double is the same as double (prec = 53).
      do {
        s += width;
        x += RandomPower2::shiftf<RealType>
          (RealType(Generator::Ran() >> (s > prec ? s - prec : 0)),
           -(s > prec ? prec : s));
      } while (s < prec);
      return x;
    }
    /**
     * See documentation for RandomCanonical::Fixed<RealType,prec>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType Fixed() throw()
    { return Fixed<RealType, std::numeric_limits<RealType>::digits>(); }
    /**
     * See documentation for RandomCanonical::Fixed<RealType,prec>().
     *
     * @return the random double.
     **********************************************************************/
    double Fixed() throw() { return Fixed<double>(); }

    /**
     * An alias for RandomCanonical::Fixed<RealType>().  Returns a random
     * number of type RealType in [0,1).
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType Real() throw()
    { return Fixed<RealType>(); }
    /**
     * An alias for RandomCanonical::Fixed().  Returns a random double in
     * [0,1).
     *
     * @return the random double.
     **********************************************************************/
    double Real() throw() { return Fixed(); }

    /**
     * Return \e i \e h with \e i in (0,2<sup><i>p</i></sup>] by rounding \e u
     * up to the next \ref fixed "fixed" real.  Result is in upper interval
     * (0,1].
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec> RealType FixedU() throw()
    { return RealType(1) - Fixed<RealType, prec>(); }
    /**
     * See documentation for RandomCanonical::FixedU<RealType,prec>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType FixedU() throw()
    { return FixedU<RealType, std::numeric_limits<RealType>::digits>(); }
    /**
     * See documentation for RandomCanonical::FixedU<RealType,prec>().
     *
     * @return the random double.
     **********************************************************************/
    double FixedU() throw() { return FixedU<double>(); }

    /**
     * Return \e i \e h with \e i in [0,2<sup><i>p</i></sup>] by rounding \e u
     * to the nearest \ref fixed "fixed" real.  Result is in nearest interval
     * [0,1].  The probability of returning interior values is <i>h</i> while
     * the probability of returning the endpoints is <i>h</i>/2.
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec> RealType FixedN() throw() {
      const RealType x = Fixed<RealType, prec>();
      return x || Boolean() ? x : RealType(1);
    }
    /**
     * See documentation for RandomCanonical::FixedN<RealType,prec>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType FixedN() throw()
    { return FixedN<RealType, std::numeric_limits<RealType>::digits>(); }
    /**
     * See documentation for RandomCanonical::FixedN<RealType,prec>().
     *
     * @return the random double.
     **********************************************************************/
    double FixedN() throw() { return FixedN<double>(); }

    /**
     * Return \e i \e h with \e i in [&minus;2<sup><i>p</i></sup>,
     * 2<sup><i>p</i></sup>] by rounding 2\e u &minus; 1 to the nearest \ref
     * fixed "fixed" real.  Result is in wide interval [&minus;1,1].  The
     * probability of returning interior values is <i>h</i>/2 while the
     * probability of returning the endpoints is <i>h</i>/4.
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec> RealType FixedW() throw() {
      // Random reals in [-1, 1].  Round random in [-1, 1] to nearest multiple
      // of 1/2^prec.  Results are of the form i/2^prec for integer i in
      // [-2^prec,2^prec].
      STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer &&
                    std::numeric_limits<RealType>::radix == 2,
                    "FixedW(): bad real type RealType");
      STATIC_ASSERT(prec > 0 && prec <= std::numeric_limits<RealType>::digits,
                    "FixedW(): invalid precision");
      RealType x = -RealType(1); // Accumulator
      int s = -1;                // How many bits so far
      do {
        s += width;
        x += RandomPower2::shiftf<RealType>
          (RealType(Generator::Ran() >> (s > prec ? s - prec : 0)),
           -(s > prec ? prec : s));
      } while (s < prec);
      return (x + RealType(1) != RealType(0)) || Boolean() ? x : RealType(1);
    }
    /**
     * See documentation for RandomCanonical::FixedW<RealType,prec>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType FixedW() throw()
    { return FixedW<RealType, std::numeric_limits<RealType>::digits>(); }
    /**
     * See documentation for RandomCanonical::FixedW<RealType,prec>().
     *
     * @return the random double.
     **********************************************************************/
    double FixedW() throw() { return FixedW<double>(); }

    /**
     * Return (<i>i</i>+1/2)\e h with \e i in [2<sup><i>p</i>&minus;1</sup>,
     * 2<sup><i>p</i>&minus;1</sup>) by rounding \e u &minus; 1/2 to nearest
     * offset \ref fixed "fixed" real.  Result is in symmetric interval
     * (&minus;1/2,1/2).
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec> RealType FixedS() throw()
    { return Fixed<RealType, prec>() -
        ( RealType(1) - RandomPower2::pow2<RealType>(-prec) ) / 2; }
    /**
     * See documentation for RandomCanonical::FixedS<RealType,prec>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType FixedS() throw()
    { return FixedS<RealType, std::numeric_limits<RealType>::digits>(); }
    /**
     * See documentation for RandomCanonical::FixedS<RealType,prec>().
     *
     * @return the random double.
     **********************************************************************/
    double FixedS() throw() { return FixedS<double>(); }

    /**
     * Return \e i \e h with \e i in (0,2<sup><i>p</i></sup>) by rounding (1
     * &minus; \e h)\e u up to next \ref fixed "fixed" real.  Result is in open
     * interval (0,1).
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec> RealType FixedO() throw() {
      // A real of type RealType in (0, 1) with precision prec
      STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer &&
                    std::numeric_limits<RealType>::radix == 2,
                    "FixedO(): bad real type RealType");
      STATIC_ASSERT(prec > 0 && prec <= std::numeric_limits<RealType>::digits,
                    "FixedO(): invalid precision");
      RealType x;
      // Loop executed 2^prec/(2^prec-1) times on average.
      do
        x = Fixed<RealType, prec>();
      while (x == 0);
      return x;
    }
    /**
     * See documentation for RandomCanonical::FixedO<RealType,prec>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType FixedO() throw()
    { return FixedO<RealType, std::numeric_limits<RealType>::digits>(); }
    /**
     * See documentation for RandomCanonical::FixedO<RealType,prec>().
     *
     * @return the random double.
     **********************************************************************/
    double FixedO() throw() { return FixedO<double>(); }

    /**
     * Return \e i \e h with \e i in [0,2<sup><i>p</i></sup>] by rounding (1 +
     * \e h)\e u down to previous \ref fixed "fixed" real.  Result is in closed
     * interval [0,1].
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec> RealType FixedC() throw() {
      // A real of type RealType in [0, 1] with precision prec
      STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer &&
                    std::numeric_limits<RealType>::radix == 2,
                    "FixedC(): bad real type RealType");
      STATIC_ASSERT(prec > 0 && prec <= std::numeric_limits<RealType>::digits,
                    "FixedC(): invalid precision");
      if (prec < width) {
        // Sample an integer in [0, n) where n = 2^prec + 1.  This uses the
        // same logic as Unsigned(n - 1).  However, unlike Unsigned, there
        // doesn't seem to be much of a penalty for the 64-bit arithmetic here
        // when result_type = unsigned long long.  Presumably this is because
        // the compiler can do some of the arithmetic.
        const result_type
          n = (result_type(1) << (prec < width ? prec : 0)) + 1,
          // Computing this instead of 2^width/n suffices, because of the form
          // of n.
          r = Generator::mask / n,
          m = r * n;
        result_type u;
        do
          u = Generator::Ran();
        while (u >= m);
        // u is rv in [0, r * n)
        return RandomPower2::shiftf<RealType>(RealType(u / r), -prec);
        // Could also special case prec < 64, using Ran64().  However the
        // general code below is faster.
      } else {                  // prec >= width
        // Synthesize a prec+1 bit random, Y, width bits at a time.  If number
        // is odd, return Fixed<RealType, prec>() (w prob 1/2); else if number
        // is zero, return 1 (w prob 1/2^(prec+1)); else repeat.  Normalizing
        // probabilities on returned results we find that Fixed<RealType,
        // prec>() is returned with prob 2^prec/(2^prec+1), and 1 is return
        // with prob 1/(2^prec+1), as required.  Loop executed twice on average
        // and so consumes 2rvs more than rvs for Fixed<RealType, prec>().  As
        // in FloatZ, do NOT try to save on calls to Ran() by using the
        // leftover bits from Fixed.
        while (true) {
          // If prec + 1 < width then mask x with (1 << prec + 1) - 1
          const result_type x = Generator::Ran(); // Low width bits of Y
          if (x & 1u)                             // Y odd?
            return Fixed<RealType, prec>(); // Prob 1/2 on each loop iteration
          if (x)
            continue;               // Y nonzero
          int s = prec + 1 - width; // Bits left to check (s >= 0)
          while (true) {
            if (s <= 0)         // We're done.  Y = 0
              // Prob 1/2^(prec+1) on each loop iteration
              return RealType(1); // We get here once every 60000 yrs (p = 64)!
            // Check the next min(s, width) bits.
            if (Generator::Ran() >> (s > width ? 0 : width - s))
              break;
            s -= width;         // Decrement s
          }
        }
      }
    }
    /**
     * See documentation for RandomCanonical::FixedC<RealType,prec>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType FixedC() throw()
    { return FixedC<RealType, std::numeric_limits<RealType>::digits>(); }
    /**
     * See documentation for RandomCanonical::FixedC<RealType,prec>().
     *
     * @return the random double.
     **********************************************************************/
    double FixedC() throw() { return FixedC<double>(); }
    ///@}

    /**
     * \name Member functions returning real floating-point numbers
     **********************************************************************/
    ///@{

    // The floating results produces results on a floating scale.  Here the
    // separation between possible results is smaller for smaller numbers.

    /**
     * In the description of the functions FloatX returning \ref floating
     * "floating-point" numbers, \e u is a random real number uniformly
     * distributed in (0, 1), \e p is the precision, and \e e is the exponent
     * range.  Each of the functions come in three variants, e.g.,
     *   - RandomCanonical::Float<RealType,p,e>() --- return \ref floating
     *     "floating-point" real of type RealType, precision \e p, and exponent
     *     range \e e;
     *   - RandomCanonical::Float<RealType>() --- as above with \e p =
     *     std::numeric_limits<RealType>::digits and \e e =
     *     - std::numeric_limits<RealType>::min_exponent;
     *   - RandomCanonical::Float() --- as above with RealType = double.
     *
     * See the \ref reals "summary" for a comparison of the functions.
     *
     * Return result is in default interval [0,1) by rounding \e u down
     * to the previous \ref floating "floating" real.
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @tparam erange the exponent range of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec, int erange> RealType Float() throw()
    { return FloatZ<RealType, prec, erange, false>(0, 0); }
    /**
     * See documentation for RandomCanonical::Float<RealType,prec,erange>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType Float() throw() {
      return Float<RealType, std::numeric_limits<RealType>::digits,
        -std::numeric_limits<RealType>::min_exponent>();
    }
    /**
     * See documentation for RandomCanonical::Float<RealType,prec,erange>().
     *
     * @return the random double.
     **********************************************************************/
    double Float() throw() { return Float<double>(); }

    /**
     * Return result is in upper interval (0,1] by round \e u up to the
     * next \ref floating "floating" real.
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @tparam erange the exponent range of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec, int erange> RealType FloatU() throw()
    { return FloatZ<RealType, prec, erange, true>(0, 0); }
    /**
     * See documentation for RandomCanonical::FloatU<RealType,prec,erange>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType FloatU() throw() {
      return FloatU<RealType, std::numeric_limits<RealType>::digits,
        -std::numeric_limits<RealType>::min_exponent>();
    }
    /**
     * See documentation for RandomCanonical::FloatU<RealType,prec,erange>().
     *
     * @return the random double.
     **********************************************************************/
    double FloatU() throw() { return FloatU<double>(); }

    /**
     * Return result is in nearest interval [0,1] by rounding \e u to
     * the nearest \ref floating "floating" real.
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @tparam erange the exponent range of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec, int erange> RealType FloatN()
      throw() {
      // Use Float or FloatU each with prob 1/2, i.e., return Boolean() ?
      // Float() : FloatU().  However, rather than use Boolean(), we pick the
      // high bit off a Ran() and pass the rest of the number to FloatZ to use.
      // This saves 1/2 a call to Ran().
      const result_type x = Generator::Ran();
      return x >> (width - 1) ?   // equivalent to Boolean()
        // Float<RealType, prec, erange>()
        FloatZ<RealType, prec, erange, false>(width - 1, x) :
        // FloatU<RealType, prec, erange>()
        FloatZ<RealType, prec, erange, true>(width - 1, x);
    }
    /**
     * See documentation for RandomCanonical::FloatN<RealType,prec,erange>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType FloatN() throw() {
      return FloatN<RealType, std::numeric_limits<RealType>::digits,
        -std::numeric_limits<RealType>::min_exponent>();
    }
    /**
     * See documentation for RandomCanonical::FloatN<RealType,prec,erange>().
     *
     * @return the random double.
     **********************************************************************/
    double FloatN() throw() { return FloatN<double>(); }

    /**
     * Return result is in wide interval [&minus;1,1], by rounding 2\e u
     * &minus; 1 to the nearest \ref floating "floating" real.
     *
     * @tparam RealType the real type of the returned random numbers.
     * @tparam prec the precision of the returned random numbers.
     * @tparam erange the exponent range of the returned random numbers.
     * @return the random result.
     **********************************************************************/
    template<typename RealType, int prec, int erange>
    RealType FloatW() throw() {
      const result_type x = Generator::Ran();
      const int y = int(x >> (width - 2));
      return (1 - (y & 2)) *    // Equiv to (Boolean() ? -1 : 1) *
        ( y & 1 ?               // equivalent to Boolean()
          // Float<RealType, prec, erange>()
          FloatZ<RealType, prec, erange, false>(width - 2, x) :
          // FloatU<RealType, prec, erange>()
          FloatZ<RealType, prec, erange, true>(width - 2, x) );
    }
    /**
     * See documentation for RandomCanonical::FloatW<RealType,prec,erange>().
     *
     * @tparam RealType the real type of the returned random numbers.
     * @return the random result with the full precision of RealType.
     **********************************************************************/
    template<typename RealType> RealType FloatW() throw() {
      return FloatW<RealType, std::numeric_limits<RealType>::digits,
        -std::numeric_limits<RealType>::min_exponent>();
    }
    /**
     * See documentation for RandomCanonical::FloatW<RealType,prec,erange>().
     *
     * @return the random double.
     **********************************************************************/
    double FloatW() throw() { return FloatW<double>(); }
    ///@}

    /**
     * \name Member functions returning booleans
     **********************************************************************/
    ///@{
    /**
     * A coin toss.  Equivalent to RandomCanonical::Integer<bool>().
     *
     * @return true with probability 1/2.
     **********************************************************************/
    bool Boolean() throw() { return Generator::Ran() & 1u; }

    /**
     * The Bernoulli distribution, true with probability \e p.  False if \e p
     * &le; 0; true if \e p &ge; 1.  Equivalent to RandomCanonical::Float() <
     * \e p, but typically faster.
     *
     * @tparam NumericType the type (integer or real) of the argument.
     * @param[in] p the probability.
     * @return true with probability \e p.
     **********************************************************************/
    template<typename NumericType> bool Prob(NumericType p) throw();

    /**
     * True with probability <i>m</i>/<i>n</i>.  False if \e m &le; 0 or \e n <
     * 0; true if \e m &ge; \e n.  With real types, Prob(\e x, \e y) is exact
     * but slower than Prob(<i>x</i>/<i>y</i>).
     *
     * @tparam NumericType the type (integer or real) of the argument.
     * @param[in] m the numerator of the probability.
     * @param[in] n the denominator of the probability.
     * @return true with probability  <i>m</i>/<i>n</i>.
     **********************************************************************/
    template<typename NumericType>
    bool Prob(NumericType m, NumericType n) throw();
    ///@}

    // Bits

    /**
     * \name Functions returning bitsets
     * These return random bits in a std::bitset.
     **********************************************************************/
    ///@{

    /**
     * Return \e nbits random bits
     *
     * @tparam nbits the number of bits in the bitset.
     * @return the random bitset.
     **********************************************************************/
    template<int nbits> std::bitset<nbits> Bits() throw();

    ///@}

    /**
     * A "global" random number generator (not thread-safe!), initialized with
     * a fixed seed [].
     **********************************************************************/
    static RANDOMLIB_EXPORT RandomCanonical Global;

  private:
    typedef RandomSeed::u32 u32;
    typedef RandomSeed::u64 u64;
    /**
     * A helper for Integer(\e n).  A random unsigned integer in [0, \e n].  If
     * \e n &ge; 2<sup>32</sup>, this \e must be invoked with \e onep = false.
     * Otherwise, it \e should be invoked with \e onep = true.
     **********************************************************************/
    template<typename UIntT>
    typename UIntT::type Unsigned(typename UIntT::type n) throw();

    /**
     * A helper for Float and FloatU.  Produces \e up ? FloatU() : Float().  On
     * entry the low \e b bits of \e m are usable random bits.
     **********************************************************************/
    template<typename RealType, int prec, int erange, bool up>
    RealType FloatZ(int b, result_type m) throw();

    /**
     * The one-argument version of Prob for real types
     **********************************************************************/
    template<typename RealType> bool ProbF(RealType z) throw();
    /**
     * The two-argument version of Prob for real types
     **********************************************************************/
    template<typename RealType> bool ProbF(RealType x, RealType y) throw();
  };

  template<class Generator>
  RandomCanonical<Generator>::RandomCanonical(seed_type n)
    : Generator(n) {
    // Compile-time checks on real types
#if HAVE_LONG_DOUBLE
    STATIC_ASSERT(std::numeric_limits<float>::radix == 2 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<long double>::radix == 2,
                  "RandomCanonical: illegal floating type");
    STATIC_ASSERT(0 <= std::numeric_limits<float>::digits &&
                  std::numeric_limits<float>::digits <=
                  std::numeric_limits<double>::digits &&
                  std::numeric_limits<double>::digits <=
                  std::numeric_limits<long double>::digits,
                  "RandomCanonical: inconsistent floating precision");
#else
    STATIC_ASSERT(std::numeric_limits<float>::radix == 2 &&
                  std::numeric_limits<double>::radix == 2,
                  "RandomCanonical: illegal floating type");
    STATIC_ASSERT(0 <= std::numeric_limits<float>::digits &&
                  std::numeric_limits<float>::digits <=
                  std::numeric_limits<double>::digits,
                  "RandomCanonical: inconsistent floating precision");
#endif
#if HAVE_LONG_DOUBLE
#endif
#if RANDOMLIB_POWERTABLE
    // checks on power2
#if HAVE_LONG_DOUBLE
    STATIC_ASSERT(std::numeric_limits<long double>::digits ==
                  RANDOMLIB_LONGDOUBLEPREC,
                  "RandomPower2: RANDOMLIB_LONGDOUBLEPREC incorrect");
#else
    STATIC_ASSERT(std::numeric_limits<double>::digits ==
                  RANDOMLIB_LONGDOUBLEPREC,
                  "RandomPower2: RANDOMLIB_LONGDOUBLEPREC incorrect");
#endif
    // Make sure table hasn't underflowed
    STATIC_ASSERT(RandomPower2::minpow >=
                  std::numeric_limits<float>::min_exponent -
                  (RANDOMLIB_HASDENORM(float) ?
                   std::numeric_limits<float>::digits : 1),
                  "RandomPower2 table underflow");
    STATIC_ASSERT(RandomPower2::maxpow >= RandomPower2::minpow + 1,
                  "RandomPower2 table empty");
    // Needed by RandomCanonical::Fixed<long double>()
#if HAVE_LONG_DOUBLE
    STATIC_ASSERT(RandomPower2::minpow <=
                  -std::numeric_limits<long double>::digits,
                  "RandomPower2 minpow not small enough for long double");
#else
    STATIC_ASSERT(RandomPower2::minpow <=
                  -std::numeric_limits<double>::digits,
                  "RandomPower2 minpow not small enough for double");
#endif
    // Needed by ProbF
    STATIC_ASSERT(RandomPower2::maxpow - width >= 0,
                  "RandomPower2 maxpow not large enough for ProbF");
#endif
    // Needed for RandomCanonical::Bits()
    STATIC_ASSERT(2 * std::numeric_limits<unsigned long>::digits - width >= 0,
                  "Bits<n>(): unsigned long too small");
  }

  template<class Generator> template<typename IntType>
  inline IntType RandomCanonical<Generator>::Integer() throw() {
    // A random integer of type IntType in [min(IntType), max(IntType)].
    STATIC_ASSERT(std::numeric_limits<IntType>::is_integer &&
                  std::numeric_limits<IntType>::radix == 2,
                  "Integer: bad integer type IntType");
    const int d = std::numeric_limits<IntType>::digits +
      std::numeric_limits<IntType>::is_signed; // Include the sign bit
    // Check that we have enough digits in Ran64
    STATIC_ASSERT(d > 0 && d <= 64, "Integer: bad bit-size");
    if (d <= width)
      return IntType(Generator::Ran());
    else                        // d <= 64
      return IntType(Generator::Ran64());
  }

  template<class Generator> template<typename UIntT>
  inline typename UIntT::type
  RandomCanonical<Generator>::Unsigned(typename UIntT::type n) throw() {
    // A random unsigned in [0, n].  In n fits in 32-bits, call with UIntType =
    // u32 and onep = true else call with UIntType = u64 and onep = false.
    // There are a few cases (e.g., n = 0x80000000) where on a 64-bit machine
    // with a 64-bit Generator it would be quicker to call this with UIntType =
    // result_type and invoke Ran().  However this speed advantage disappears
    // if the argument isn't a compile time constant.
    //
    // Special case n == 0 is handled by the callers of Unsigned.  The
    // following is to guard against a division by 0 in the return statement
    // (but it shouldn't happen).
    n = n ? n : 1U;             // n >= 1
    // n1 = n + 1, but replace overflowed value by 1.  Overflow occurs, e.g.,
    // when n = u32::mask and then we have r1 = 0, m = u32::mask.
    const typename UIntT::type n1 = ~n ? n + 1U : 1U;
    // "Ratio method".  Find m = r * n1 - 1, s.t., 0 < (q - n1) < m <= q, where
    // q = max(UIntType), and sample in u in [0, m] and return u / r.  If onep
    // then we use Ran32() else Rand64().
    const typename UIntT::type
      // r = floor((q + 1)/n1), r1 = r - 1, avoiding overflow.  Actually
      // overflow can occur if std::numeric_limits<u32>::digits == 64, because
      // then we can have onep && n > U32_MASK.  This is however ruled out by
      // the callers to Unsigned.  (If Unsigned is called in this way, the
      // results are bogus, but there is no error condition.)
      r1 = ((UIntT::width == 32 ? typename UIntT::type(u32::mask) :
             typename UIntT::type(u64::mask)) - n) / n1,
      m = r1 * n1 + n;          // m = r * n1 - 1, avoiding overflow
    // Here r1 in [0, (q-1)/2], m in [(q+1)/2, q]
    typename UIntT::type u;     // Find a random number in [0, m]
    do
      // For small n1, this is executed once (since m is nearly q).  In the
      // worst case the loop is executed slightly less than twice on average.
      u = UIntT::width == 32 ? typename UIntT::type(Generator::Ran32()) :
        typename UIntT::type(Generator::Ran64());
    while (u > m);
    // Now u is in [0, m] = [0, r * n1), so u / r is in [0, n1) = [0, n].  An
    // alternative unbiased method would be u % n1; but / appears to be faster.
    return u / (r1 + 1U);
  }

  template<class Generator> template<typename IntType>
  inline IntType RandomCanonical<Generator>::Integer(IntType n) throw() {
    // A random integer of type IntType in [0, n).  If n == 0, treat as
    // max(IntType) + 1.  If n < 0, treat as 1 and return 0.
    // N.B. Integer<IntType>(0) is equivalent to Integer<IntType>() for
    // unsigned types.  For signed types, the former returns a non-negative
    // result and the latter returns a result in the full range.
    STATIC_ASSERT(std::numeric_limits<IntType>::is_integer &&
                  std::numeric_limits<IntType>::radix == 2,
                  "Integer(n): bad integer type IntType");
    const int d = std::numeric_limits<IntType>::digits;
    // Check that we have enough digits in Ran64
    STATIC_ASSERT(d > 0 && d <= 64, "Integer(n): bad bit-size");
    return n > IntType(1) ?
      (d <= 32 || n - 1 <= IntType(u32::mask) ?
       IntType(Unsigned<u32>(u32::type(n - 1))) :
       IntType(Unsigned<u64>(u64::type(n - 1)))) :
      ( n ? IntType(0) :        // n == 1 || n < 0
        Integer<IntType, d>()); // n == 0
  }

  template<class Generator> template<typename IntType>
  inline IntType RandomCanonical<Generator>::IntegerC(IntType n) throw() {
    // A random integer of type IntType in [0, n]
    STATIC_ASSERT(std::numeric_limits<IntType>::is_integer &&
                  std::numeric_limits<IntType>::radix == 2,
                  "IntegerC(n): bad integer type IntType");
    const int d = std::numeric_limits<IntType>::digits;
    // Check that we have enough digits in Ran64
    STATIC_ASSERT(d > 0 && d <= 64, "IntegerC(n): bad bit-size");
    return n > IntType(0) ?
      (d <= 32 || n <= IntType(u32::mask) ?
       IntType(Unsigned<u32>(u32::type(n))) :
       IntType(Unsigned<u64>(u64::type(n))))
      : IntType(0);             // n <= 0
  }

  template<class Generator> template<typename IntType>
  inline IntType RandomCanonical<Generator>::IntegerC(IntType m, IntType n)
    throw() {
    // A random integer of type IntType in [m, n]
    STATIC_ASSERT(std::numeric_limits<IntType>::is_integer &&
                  std::numeric_limits<IntType>::radix == 2,
                  "IntegerC(m,n): bad integer type IntType");
    const int d = std::numeric_limits<IntType>::digits +
      std::numeric_limits<IntType>::is_signed; // Include sign bit
    // Check that we have enough digits in Ran64
    STATIC_ASSERT(d > 0 && d <= 64, "IntegerC(m,n): bad bit-size");
    // The unsigned subtraction, n - m, avoids the underflow that is possible
    // in the signed operation.
    return m + (n <= m ? 0 :
                d <= 32 ?
                IntType(IntegerC<u32::type>(u32::type(n) - u32::type(m))) :
                IntType(IntegerC<u64::type>(u64::type(n) - u64::type(m))));
  }

  template<class Generator>
  template<typename RealType, int prec, int erange, bool up> inline
  RealType RandomCanonical<Generator>::FloatZ(int b, result_type m) throw() {
    // Produce up ? FloatU() : Float().  On entry the low b bits of m are
    // usable random bits.
    STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer &&
                  std::numeric_limits<RealType>::radix == 2,
                  "FloatZ: bad real type RealType");
    STATIC_ASSERT(prec > 0 && prec <= std::numeric_limits<RealType>::digits,
                  "FloatZ: invalid precision");
    STATIC_ASSERT(erange >= 0, "FloatZ: invalid exponent range");
    // With subnormals: condition that smallest number is representable
    STATIC_ASSERT(!RANDOMLIB_HASDENORM(RealType) ||
                  // Need 1/2^(erange+prec) > 0
                  prec + erange <= std::numeric_limits<RealType>::digits -
                  std::numeric_limits<RealType>::min_exponent,
                  "FloatZ: smallest number cannot be represented");
    // Without subnormals :condition for no underflow in while loop
    STATIC_ASSERT(RANDOMLIB_HASDENORM(RealType) ||
                  // Need 1/2^(erange+1) > 0
                  erange <= - std::numeric_limits<RealType>::min_exponent,
                  "FloatZ: underflow possible");

    // Simpler (but slower) version of FloatZ.  However this method cannot
    // handle the full range of exponents and, in addition, is slower on
    // average.
    // template<typename RealType, int prec, int erange, bool up>
    // RealType FloatZ() {
    //   RealType x = Fixed<RealType, erange + 1>();
    //   int s;         // Determine exponent (-erange <= s <= 0)
    //   frexp(x, &s);      // Prob(s) = 2^(s-1)
    //   // Scale number in [1,2) by 2^(s-1).  If x == 0 scale number in [0,1).
    //   return ((up ? FixedU<RealType, prec - 1>() :
    //            Fixed<RealType, prec - 1>()) + (x ? 1 : 0)) *
    //     RandomPower2::pow2<RealType>(s - 1);
    // }
    //
    // Use {a, b} to denote the inteval: up ? (a, b] : [a, b)
    //
    // The code produces the number as
    //
    // Interval             count       prob = spacing
    // {1,2} / 2            2^(prec-1)  1/2^prec
    // {1,2} / 2^s          2^(prec-1)  1/2^(prec+s-1)      for s = 2..erange+1
    // {0,1} / 2^(erange+1) 2^(prec-1)  1/2^(prec+erange)

    // Generate prec bits in {0, 1}
    RealType x = up ? FixedU<RealType, prec>() : Fixed<RealType, prec>();
    // Use whole interval if erange == 0 and handle the interval {1/2, 1}
    if (erange == 0 || (up ? x > RealType(0.5) : x >= RealType(0.5)))
      return x;
    x += RealType(0.5);         // Shift remaining portion to {1/2, 1}
    if (b == 0) {
      m = Generator::Ran();     // Random bits
      b = width;                // Bits available in m
    }
    int sm = erange;            // sm = erange - s + 2
    // Here x in {1, 2} / 2, prob 1/2
    do {                        // s = 2 thru erange+1, sm = erange thru 1
      x /= 2;
      if (m & 1u)
        return x;               // x in {1, 2} / 2^s, prob 1/2^s
      if (--b)
        m >>= 1;
      else {
        m = Generator::Ran();
        b = width;
      }
    } while (--sm);
    // x in {1, 2} / 2^(erange+1), prob 1/2^(erange+1).  Don't worry about the
    // possible overhead of the calls to pow here.  We rarely get here.
    if (RANDOMLIB_HASDENORM(RealType) || // subnormals allowed
        // No subnormals but smallest number still representable
        prec + erange <= -std::numeric_limits<RealType>::min_exponent + 1 ||
        // Possibility of underflow, so have to test on x.  Here, we have -prec
        // + 1 < erange + min_exp <= 0 so pow2 can be used
        x >= (RealType(1) +
              RandomPower2::pow2<RealType>
              (erange + std::numeric_limits<RealType>::min_exponent)) *
        (erange + 1 > -RandomPower2::minpow ?
         std::pow(RealType(2), - erange - 1) :
         RandomPower2::pow2<RealType>(- erange - 1)))
      // shift x to {0, 1} / 2^(erange+1)
      // Use product of pow's since max(erange + 1) =
      // std::numeric_limits<RealType>::digits -
      // std::numeric_limits<RealType>::min_exponent and pow may underflow
      return x -
        (erange + 1 > -RandomPower2::minpow ?
         std::pow(RealType(2), -(erange + 1)/2) *
         std::pow(RealType(2), -(erange + 1) + (erange + 1)/2) :
         RandomPower2::pow2<RealType>(- erange - 1));
    else
      return up ?               // Underflow to up ? min() : 0
        // pow is OK here.
        std::pow(RealType(2), std::numeric_limits<RealType>::min_exponent - 1)
        : RealType(0);
  }

  /// \cond SKIP
  // True with probability n.  Since n is an integer this is equivalent to n >
  // 0.
  template<class Generator> template<typename IntType>
  inline bool RandomCanonical<Generator>::Prob(IntType n) throw() {
    STATIC_ASSERT(std::numeric_limits<IntType>::is_integer,
                  "Prob(n): invalid integer type IntType");
    return n > 0;
  }
  /// \endcond

  // True with probability p.  true if p >= 1, false if p <= 0 or isnan(p).
  template<class Generator> template<typename RealType>
  inline bool RandomCanonical<Generator>::ProbF(RealType p) throw() {
    // Simulate Float<RealType>() < p.  The definition involves < (instead of
    // <=) because Float<RealType>() is in [0,1) so it is "biased downwards".
    // Instead of calling Float<RealType>(), we generate only as many bits as
    // necessary to determine the result.  This makes the routine considerably
    // faster than Float<RealType>() < x even for type float.  Compared with
    // the inexact Fixed<RealType>() < p, this is about 20% slower with floats
    // and 20% faster with doubles and long doubles.
    STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer &&
                  std::numeric_limits<RealType>::radix == 2,
                  "ProbF(p): invalid real type RealType");
    // Generate Real() c bits at a time where c is chosen so that cast doesn't
    // loose any bits and so that it uses up just one rv.
    const int c = std::numeric_limits<RealType>::digits > width ?
      width : std::numeric_limits<RealType>::digits;
    STATIC_ASSERT(c > 0, "ProbF(p): Illegal chunk size");
    const RealType mult = RandomPower2::pow2<RealType>(c);
    // A recursive definition:
    //
    // return p > RealType(0) &&
    //   (p >= RealType(1) ||
    //    ProbF(mult * p - RealType(Integer<result_type, c>())));
    //
    // Pre-loop tests needed to avoid overflow
    if (!(p > RealType(0)))     // Ensure false if isnan(p)
      return false;
    else if (p >= RealType(1))
      return true;
    do {                        // Loop executed slightly more than once.
      // Here p is in (0,1).  Write Fixed() = (X + y)/mult where X is an
      // integer in [0, mult) and y is a real in [0,1).  Then Fixed() < p
      // becomes p' > y where p' = p * mult - X.
      p *= mult;                // Form p'.  Multiplication is exact
      p -= RealType(Integer<result_type, c>()); // Also exact
      if (p <= RealType(0))
        return false;           // If p' <= 0 the result is definitely false.
      // Exit if p' >= 1; the result is definitely true.  Otherwise p' is in
      // (0,1) and the result is true with probability p'.
    } while (p < RealType(1));
    return true;
  }

  /// \cond SKIP
  // True with probability m/n (ratio of integers)
  template<class Generator> template<typename IntType>
  inline bool RandomCanonical<Generator>::Prob(IntType m, IntType n) throw() {
    STATIC_ASSERT(std::numeric_limits<IntType>::is_integer,
                  "Prob(m,n): invalid integer type IntType");
    // Test n >= 0 without triggering compiler warning when n = unsigned
    return m > 0 && (n > 0 || n == 0) && (m >= n || Integer<IntType>(n) < m);
  }
  /// \endcond

  // True with probability x/y (ratio of reals)
  template<class Generator> template<typename RealType>
  inline bool RandomCanonical<Generator>::ProbF(RealType x, RealType y)
    throw() {
    STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer &&
                  std::numeric_limits<RealType>::radix == 2,
                  "ProbF(x,y): invalid real type RealType");
    if (!(x > RealType(0) && y >= RealType(0))) // Do the trivial cases
      return false;             // Also if either x or y is a nan
    else if (x >= y)
      return true;
    // Now 0 < x < y
    int ex, ey;                 // Extract exponents
    x = std::frexp(x, &ex);
    y = std::frexp(y, &ey);
    // Now 0.5 <= x,y < 1
    if (x > y) {
      x *= RealType(0.5);
      ++ex;
    }
    int s = ey - ex;
    // Now 0.25 < x < y < 1, s >= 0, 0.5 < x/y <= 1
    // Return true with prob 2^-s * x/y
    while (s > 0) {             // With prob 1 - 2^-s return false
      // Check the next min(s, width) bits.
      if (Generator::Ran() >> (s > width ? 0 : width - s))
        return false;
      s -= width;
    }
    // Here with prob 2^-s
    const int c = std::numeric_limits<RealType>::digits > width ?
      width : std::numeric_limits<RealType>::digits;
    STATIC_ASSERT(c > 0, "ProbF(x,y): invalid chunk size");
    const RealType mult = RandomPower2::pow2<RealType>(c);
    // Generate infinite precision z = Real().
    // As soon as we know z > y, start again
    // As soon as we know z < x, return true
    // As soon as we know x < z < y, return false
    while (true) {              // Loop executed 1/y on average
      RealType xa = x, ya = y;
      while (true) {            // Loop executed slightly more than once
        // xa <= ya, ya > 0, xa < 1.
        // Here (xa,ya) are in (0,1).  Write z = (Z + z')/mult where Z is an
        // integer in [0, mult) and z' is a real in [0,1).  Then z < x becomes
        // z' < x' where x' = x * mult - Z.
        const RealType d = RealType(Integer<result_type, c>());
        if (ya < RealType(1)) {
          ya *= mult;           // Form ya'
          ya -= d;
          if (ya <= RealType(0))
            break;              // z > y, start again
        }
        if (xa > RealType(0)) {
          xa *= mult;           // Form xa'
          xa -= d;
          if (xa >= RealType(1))
            return true;        // z < x
        }
        if (xa <= RealType(0) && ya >= RealType(1))
          return false;         // x < z < y
      }
    }
  }

  template<class Generator> template<int nbits>
  inline std::bitset<nbits> RandomCanonical<Generator>::Bits() throw() {
    // Return nbits random bits
    STATIC_ASSERT(nbits >= 0, "Bits<n>(): invalid nbits");
    const int ulbits = std::numeric_limits<bitset_uint_t>::digits;
    STATIC_ASSERT(2 * ulbits >= width,
                  "Bits<n>(): integer constructor type too narrow");
    std::bitset<nbits> b;
    int m = nbits;

    while (m > 0) {
      result_type x = Generator::Ran();
      if (m < nbits)
        b <<= (width > ulbits ? width - ulbits : width);
      if (width > ulbits &&     // x doesn't fit into a bitset_uint_t
          // But on the first time through the loop the most significant bits
          // may not be needed.
          (nbits > ((nbits-1)/width) * width + ulbits || m < nbits)) {
        // Handle most significant width - ulbits bits.
        b |= (bitset_uint_t)(x >> (width > ulbits ? ulbits : 0));
        b <<= ulbits;
      }
      // Bitsets can be constructed from a bitset_uint_t.
      b |= (bitset_uint_t)(x);
      m -= width;
    }
    return b;
  }
  /// \cond SKIP

  // The specialization of Integer<bool> is required because bool(int) in the
  // template definition will test for non-zeroness instead of returning the
  // low bit.
#if HAVE_LONG_DOUBLE
#define RANDOMCANONICAL_SPECIALIZE(RandomType)              \
  template<> template<>                                     \
  inline bool RandomType::Integer<bool>()                   \
    throw() { return Boolean(); }                           \
  RANDOMCANONICAL_SPECIALIZE_PROB(RandomType, float)        \
  RANDOMCANONICAL_SPECIALIZE_PROB(RandomType, double)       \
  RANDOMCANONICAL_SPECIALIZE_PROB(RandomType, long double)
#else
#define RANDOMCANONICAL_SPECIALIZE(RandomType)              \
  template<> template<>                                     \
  inline bool RandomType::Integer<bool>()                   \
    throw() { return Boolean(); }                           \
  RANDOMCANONICAL_SPECIALIZE_PROB(RandomType, float)        \
  RANDOMCANONICAL_SPECIALIZE_PROB(RandomType, double)
#endif

  // Connect Prob(p) with ProbF(p) for all real types
  // Connect Prob(x, y) with ProbF(x, y) for all real types
#define RANDOMCANONICAL_SPECIALIZE_PROB(RandomType, RealType)       \
  template<> template<>                                             \
  inline bool RandomType::Prob<RealType>(RealType p)                \
    throw() { return ProbF<RealType>(p); }                          \
  template<> template<>                                             \
  inline bool RandomType::Prob<RealType>(RealType x, RealType y)    \
    throw() { return ProbF<RealType>(x, y); }

  RANDOMCANONICAL_SPECIALIZE(RandomCanonical<MRandomGenerator32>)
  RANDOMCANONICAL_SPECIALIZE(RandomCanonical<MRandomGenerator64>)
  RANDOMCANONICAL_SPECIALIZE(RandomCanonical<SRandomGenerator32>)
  RANDOMCANONICAL_SPECIALIZE(RandomCanonical<SRandomGenerator64>)

#undef RANDOMCANONICAL_SPECIALIZE
#undef RANDOMCANONICAL_SPECIALIZE_PROB

  /// \endcond

  /**
   * Hook XRandomNN to XRandomGeneratorNN
   **********************************************************************/
  typedef RandomCanonical<MRandomGenerator32> MRandom32;
  typedef RandomCanonical<MRandomGenerator64> MRandom64;
  typedef RandomCanonical<SRandomGenerator32> SRandom32;
  typedef RandomCanonical<SRandomGenerator64> SRandom64;

} // namespace RandomLib

namespace std {

  /**
   * Swap two RandomCanonicals.  This is about 3x faster than the default swap.
   **********************************************************************/
  template<class Generator>
  void swap(RandomLib::RandomCanonical<Generator>& r,
            RandomLib::RandomCanonical<Generator>& s) throw() {
    r.swap(s);
  }

} // namespace srd

#if defined(_MSC_VER)
#  pragma warning (pop)
#endif

#endif  // RANDOMLIB_RANDOMCANONICAL_HPP
