/**
 * \file RandomAlgorithm.hpp
 * \brief Header for MT19937 and SFMT19937.
 *
 * This provides an interface to the Mersenne Twister
 * <a href="http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html">
 * MT19937</a> and SIMD oriented Fast Mersenne Twister
 * <a href="http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/index.html">
 * SFMT19937</a> random number engines.
 *
 * Interface routines written by Charles Karney <charles@karney.com> and
 * licensed under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_RANDOMALGORITHM_HPP)
#define RANDOMLIB_RANDOMALGORITHM_HPP 1

#include <RandomLib/RandomType.hpp>
#include <stdexcept>
#include <string>
#if defined(HAVE_SSE2) && HAVE_SSE2
#include <emmintrin.h>
#endif

#if (defined(HAVE_SSE2) && HAVE_SSE2) && (defined(HAVE_ALTIVEC) && HAVE_ALTIVEC)
#error "HAVE_SSE2 and HAVE_ALTIVEC should not both be defined"
#endif

#if defined(_MSC_VER)
// Squelch warnings about casts truncating constants
#  pragma warning (push)
#  pragma warning (disable: 4310)
#endif

namespace RandomLib {

  /**
   * \brief The %MT19937 random number engine.
   *
   * This provides an interface to Mersenne Twister random number engine,
   * <a href="http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html">
   * MT19937</a>.  See\n Makoto Matsumoto and Takuji Nishimura,\n Mersenne
   * Twister: A 623-Dimensionally Equidistributed Uniform Pseudo-Random Number
   * Generator,\n ACM TOMACS 8, 3--30 (1998)
   *
   * This is adapted from the 32-bit and 64-bit C versions available at
   * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html and
   * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt64.html
   *
   * The template argument give the type \e RandomType of the "natural" result.
   * This incorporates the bit width and the C++ type of the result.  Although
   * the two versions of MT19937 produce different sequences, the
   * implementations here are portable across 32-bit and 64-bit architectures.
   *
   * The class chiefly supplies the method for advancing the state by
   * Transition.
   *
   * @tparam RandomType the type of the results, either Random_u32 or
   *   Random_u64.
   *
   * Interface routines written by Charles Karney <charles@karney.com> and
   * licensed under the MIT/X11 License.  For more information, see
   * http://randomlib.sourceforge.net/
   **********************************************************************/
  template<class RandomType> class RANDOMLIB_EXPORT MT19937 {
  public:
    /**
     * The result RandomType
     **********************************************************************/
    typedef RandomType engine_t;
    /**
     * The internal numeric type for MT19337::Transition
     **********************************************************************/
    typedef typename engine_t::type internal_type;
  private:
    /**
     * The unsigned type of engine_t
     **********************************************************************/
    typedef typename engine_t::type engine_type;
    /**
     * The width of the engine_t
     **********************************************************************/
    static const unsigned width = engine_t::width;
    enum {
      /**
       * The Mersenne prime is 2<sup><i>P</i></sup> &minus; 1
       **********************************************************************/
      P = 19937,
      /**
       * The short lag for MT19937
       **********************************************************************/
      M = width == 32 ? 397 : 156,
      /**
       * The number of ignored bits in the first word of the state
       **********************************************************************/
      R = ((P + width - 1)/width) * width - P
    };
    static const engine_type mask = engine_t::mask;
    /**
     * Magic matrix for MT19937
     **********************************************************************/
    static const engine_type magic =
      width == 32 ? 0x9908b0dfULL : 0xb5026f5aa96619e9ULL;
    /**
     * Mask for top \e width &minus; \e R bits of a word
     **********************************************************************/
    static const engine_type upper = mask << R & mask;
    /**
     * Mask for low \e R bits of a <i>width</i>-bit word
     **********************************************************************/
    static const engine_type lower = ~upper & mask;

  public:
    /**
     * A version number "EnMT" or "EnMU" to ensure safety of Save/Load.  This
     * needs to be unique across RandomAlgorithms.
     **********************************************************************/
    static const unsigned version = 0x456e4d54UL + (engine_t::width/32 - 1);
    enum {
      /**
       * The size of the state.  This is the long lag for MT19937.
       **********************************************************************/
      N = (P + width - 1)/width
    };
    /**
     * Advance state by \e count batches.  For speed all \e N words of state
     * are advanced together.  If \e count is negative, the state is stepped
     * backwards.  This is the meat of the MT19937 engine.
     *
     * @param[in] count how many batches to advance.
     * @param[in,out] statev the internal state of the random number generator.
     **********************************************************************/
    static void Transition(long long count, internal_type statev[]) throw();

    /**
     * Manipulate a word of the state prior to output.
     *
     * @param[in] y a word of the state.
     * @return the result.
     **********************************************************************/
    static engine_type Generate(engine_type y) throw();

    /**
     * Convert an arbitrary state into a legal one.  This consists of (a)
     * turning on one bit if the state is all zero and (b) making 31 bits of
     * the state consistent with the other 19937 bits.
     *
     * @param[in,out] state the state of the generator.
     **********************************************************************/
    static void NormalizeState(engine_type state[]) throw();

    /**
     * Check that the state is legal, throwing an exception if it is not.  At
     * the same time, accumulate a checksum of the state.
     *
     * @param[in] state the state of the generator.
     * @param[in,out] check an accumulated checksum.
     **********************************************************************/
    static void CheckState(const engine_type state[], Random_u32::type& check);

    /**
     * Return the name of the engine
     *
     * @return the name.
     **********************************************************************/
    static std::string Name() throw() {
      return "MT19937<Random_u" + std::string(width == 32 ? "32" : "64") + ">";
    }
  };

  /// \cond SKIP
  template<>
  inline Random_u32::type MT19937<Random_u32>::Generate(engine_type y) throw() {
    y ^= y >> 11;
    y ^= y <<  7 & engine_type(0x9d2c5680UL);
    y ^= y << 15 & engine_type(0xefc60000UL);
    y ^= y >> 18;

    return y;
  }

  template<>
  inline Random_u64::type MT19937<Random_u64>::Generate(engine_type y) throw() {
    // Specific tempering instantiation for width = 64 given in
    // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt64.html
    y ^= y >> 29 & engine_type(0x5555555555555555ULL);
    y ^= y << 17 & engine_type(0x71d67fffeda60000ULL);
    y ^= y << 37 & engine_type(0xfff7eee000000000ULL);
    y ^= y >> 43;

    return y;
  }
  /// \endcond

  /**
   * \brief The SFMT random number engine.
   *
   * This provides an implementation of the SIMD-oriented Fast Mersenne Twister
   * random number engine,
   * <a href="http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/index.html">
   * SFMT</a>.  See\n Mutsuo Saito,\n An Application of Finite Field: Design
   * and Implementation of 128-bit Instruction-Based Fast Pseudorandom Number
   * Generator,\n Master's Thesis, Dept. of Math., Hiroshima University
   * (Feb. 2007).\n
   * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/M062821.pdf
   * Mutsuo Saito and Makoto Matsumoto,\n
   * SIMD-oriented Fast Mersenne Twister: a 128-bit Pseudorandom Number
   * Generator,\n accepted in the proceedings of MCQMC2006\n
   * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/ARTICLES/sfmt.pdf
   *
   * The template argument gives the type \e RandomType of the "natural"
   * result.  This incorporates the bit width and the C++ type of the result.
   * The 32-bit and 64-bit versions of SFMT19937 produce the same sequences and
   * the differing only in whether how the state is represented.  The
   * implementation includes a version using 128-bit SSE2 instructions.  On
   * machines without these instructions, portable implementations using
   * traditional operations are provided.  With the same starting seed,
   * SRandom32::Ran64() and SRandom64::Ran64() produces the same sequences.
   * Similarly SRandom64::Ran32() produces every other member of the sequence
   * produced by SRandom32::Ran32().
   *
   * The class chiefly supplies the method for advancing the state by
   * Transition.
   *
   * @tparam RandomType the type of the results, either Random_u32 or
   *   Random_u64.
   *
   * Written by Charles Karney <charles@karney.com> and licensed under the
   * MIT/X11 License.  For more information, see
   * http://randomlib.sourceforge.net/
   **********************************************************************/
  template<class RandomType> class RANDOMLIB_EXPORT SFMT19937 {
  public:
    /**
     * The result RandomType
     **********************************************************************/
    typedef RandomType engine_t;
#if defined(HAVE_SSE2) && HAVE_SSE2
    typedef __m128i internal_type;
#elif defined(HAVE_ALTIVEC) && HAVE_ALTIVEC
    typedef vector unsigned internal_type;
#else
    /**
     * The internal numeric type for SFMT19337::Transition
     **********************************************************************/
    typedef typename engine_t::type internal_type;
#endif
  private:
    /**
     * The unsigned type of engine_t
     **********************************************************************/
    typedef typename engine_t::type engine_type;
    /**
     * The width of the engine_t
     **********************************************************************/
    static const unsigned width = engine_t::width;
    enum {
      /**
       * The Mersenne prime is 2<sup><i>P</i></sup> &minus; 1
       **********************************************************************/
      P = 19937,
      /**
       * The long lag for SFMT19937 in units of 128-bit words
       **********************************************************************/
      N128 = (P + 128 - 1)/128,
      /**
       * How many width words per 128-bit word.
       **********************************************************************/
      R = 128 / width,
      /**
       * The short lag for SFMT19937  in units of 128-bit words
       **********************************************************************/
      M128 = 122,
      /**
       * The short lag for SFMT19937
       **********************************************************************/
      M = M128 * R
    };
#if (defined(HAVE_SSE2) && HAVE_SSE2) || (defined(HAVE_ALTIVEC) && HAVE_ALTIVEC)
    static const Random_u32::type magic0 = 0x1fffefUL;
    static const Random_u32::type magic1 = 0x1ecb7fUL;
    static const Random_u32::type magic2 = 0x1affffUL;
    static const Random_u32::type magic3 = 0x1ffff6UL;
#else
    /**
     * Magic matrix for SFMT19937.  Only the low 21 (= 32 &minus; 11) bits need
     * to be set.  (11 is the right shift applied to the words before masking.
     **********************************************************************/
    static const engine_type
      magic0 = width == 32 ? 0x1fffefULL : 0x1ecb7f001fffefULL;
    static const engine_type
      magic1 = width == 32 ? 0x1ecb7fULL : 0x1ffff6001affffULL;
    static const engine_type
      magic2 = width == 32 ? 0x1affffULL :                0ULL;
    static const engine_type
      magic3 = width == 32 ? 0x1ffff6ULL :                0ULL;
#endif
    /**
     * Mask for simulating u32 << 18 with 64-bit words
     **********************************************************************/
    static const engine_type mask18 = engine_type(0xfffc0000fffc0000ULL);
    /**
     * Magic constants needed by "period certification"
     **********************************************************************/
    static const engine_type PARITY0 = 1U;
    static const engine_type PARITY1 = width == 32 ? 0U : 0x13c9e68400000000ULL;
    static const engine_type PARITY2 = 0U;
    static const engine_type PARITY3 = width == 32 ? 0x13c9e684UL : 0U;
    /**
     * Least significant bit of PARITY
     **********************************************************************/
    static const unsigned PARITY_LSB = 0;
    static const engine_type mask = engine_t::mask;

  public:
    /**
     * A version number "EnSM" or "EnSN" to ensure safety of Save/Load.  This
     * needs to be unique across RandomAlgorithms.
     **********************************************************************/
    static const unsigned version = 0x456e534dUL + (engine_t::width/32 - 1);
    enum {
      /**
       * The size of the state.  The long lag for SFMT19937
       **********************************************************************/
      N = N128 * R
    };
    /**
     * Advance state by \e count batches.  For speed all \e N words of state
     * are advanced together.  If \e count is negative, the state is stepped
     * backwards. This is the meat of the SFMT19937 engine.
     *
     * @param[in] count how many batches to advance.
     * @param[in,out] statev the internal state of the random number generator.
     **********************************************************************/
    static void Transition(long long count, internal_type statev[])
      throw();

    /**
     * Manipulate a word of the state prior to output.  This is a no-op for
     * SFMT19937.
     *
     * @param[in] y a word of the state.
     * @return the result.
     **********************************************************************/
    static engine_type Generate(engine_type y) throw() { return y; }

    /**
     * Convert an arbitrary state into a legal one.  This consists a "period
     * certification to ensure that the period of the generator is at least
     * 2<sup><i>P</i></sup> &minus; 1.
     *
     * @param[in,out] state the state of the generator.
     **********************************************************************/
    static void NormalizeState(engine_type state[]) throw();

    /**
     * Check that the state is legal, throwing an exception if it is not.  This
     * merely verifies that the state is not all zero.  At the same time,
     * accumulate a checksum of the state.
     *
     * @param[in] state the state of the generator.
     * @param[in,out] check an accumulated checksum.
     **********************************************************************/
    static void CheckState(const engine_type state[], Random_u32::type& check);

    /**
     * Return the name of the engine
     *
     * @return the name.
     **********************************************************************/
    static std::string Name() throw() {
      return "SFMT19937<Random_u" +
        std::string(width == 32 ? "32" : "64") + ">";
    }
  };

} // namespace RandomLib

#if defined(_MSC_VER)
#  pragma warning (pop)
#endif

#endif  // RANDOMLIB_RANDOMALGORITHM_HPP
