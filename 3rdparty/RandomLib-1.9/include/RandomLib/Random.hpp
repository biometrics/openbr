/**
 * \file Random.hpp
 * \brief Header for Random, RandomGenerator.
 *
 * This loads up the header for RandomCanonical, RandomEngine, etc., to
 * provide access to random integers of various sizes, random reals with
 * various precisions, a random probability, etc.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_RANDOM_HPP)
#define RANDOMLIB_RANDOM_HPP 1

#include <RandomLib/Config.h>

#if defined(_MSC_VER)
typedef unsigned uint32_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif

/**
 * Use table, Power2::power2, for pow2?  This isn't necessary with g++ 4.0
 * because calls to std::pow are optimized.  g++ 4.1 seems to have lost this
 * capability though!  And it's back in g++ 4.4.  So, for simplicity, assume
 * that all "current" versions of g++ perform the optimization.
 **********************************************************************/
#if !defined(RANDOMLIB_POWERTABLE)
#if defined(__GNUC__)
#define RANDOMLIB_POWERTABLE 0
#else
// otherwise use a lookup table
#define RANDOMLIB_POWERTABLE 1
#endif
#endif

#if !HAVE_LONG_DOUBLE || defined(_MSC_VER)
#define RANDOMLIB_LONGDOUBLEPREC 53
#elif defined(__sparc)
#define RANDOMLIB_LONGDOUBLEPREC 113
#else
/**
 * The precision of long doubles, used for sizing Power2::power2.  64 on
 * Linux/Intel, 106 on MaxOS/PowerPC
 **********************************************************************/
#define RANDOMLIB_LONGDOUBLEPREC __LDBL_MANT_DIG__
#endif

/**
 * A compile-time assert.  Use C++11 static_assert, if available.
 **********************************************************************/
#if !defined(STATIC_ASSERT)
#  if __cplusplus >= 201103
#    define STATIC_ASSERT static_assert
#  elif defined(__GXX_EXPERIMENTAL_CXX0X__)
#    define STATIC_ASSERT static_assert
#  elif defined(_MSC_VER) && _MSC_VER >= 1600
#    define STATIC_ASSERT static_assert
#  else
#    define STATIC_ASSERT(cond,reason) \
            { enum{ STATIC_ASSERT_ENUM = 1/int(cond) }; }
#  endif
#endif

/**
 * Are denormalized reals of type RealType supported?
 **********************************************************************/
#define RANDOMLIB_HASDENORM(RealType) 1

#if defined(_MSC_VER) && defined(RANDOMLIB_SHARED_LIB) && RANDOMLIB_SHARED_LIB
#  if RANDOMLIB_SHARED_LIB > 1
#    error RANDOMLIB_SHARED_LIB must be 0 or 1
#  elif defined(RandomLib_EXPORTS)
#    define RANDOMLIB_EXPORT __declspec(dllexport)
#  else
#    define RANDOMLIB_EXPORT __declspec(dllimport)
#  endif
#else
#  define RANDOMLIB_EXPORT
#endif

#include <stdexcept>

/**
 * \brief Namespace for %RandomLib
 *
 * All of %RandomLib is defined within the RandomLib namespace.  In addtiion
 * all the header files are included via %RandomLib/filename.  This minimizes
 * the likelihood of conflicts with other packages.
 **********************************************************************/
namespace RandomLib {

  /**
   * \brief Exception handling for %RandomLib
   *
   * A class to handle exceptions.  It's derived from std::runtime_error so it
   * can be caught by the usual catch clauses.
   **********************************************************************/
  class RandomErr : public std::runtime_error {
  public:

    /**
     * Constructor
     *
     * @param[in] msg a string message, which is accessible in the catch
     *   clause, via what().
     **********************************************************************/
    RandomErr(const std::string& msg) : std::runtime_error(msg) {}
  };

} // namespace RandomLib

#include <RandomLib/RandomCanonical.hpp>

#if !defined(RANDOMLIB_BUILDING_LIBRARY)

namespace RandomLib {

#if !defined(RANDOMLIB_DEFAULT_GENERATOR)
#define RANDOMLIB_DEFAULT_GENERATOR SRandomGenerator32
#endif

  /**
   * Point Random to one of a specific MT19937 generators.
   **********************************************************************/
  typedef RANDOMLIB_DEFAULT_GENERATOR RandomGenerator;

  /**
   * Hook Random to RandomGenerator
   **********************************************************************/
  typedef RandomCanonical<RandomGenerator> Random;

} // namespace RandomLib

#endif  // !defined(RANDOMLIB_BUILDING_LIBRARY)

#endif  // RANDOMLIB_RANDOM_HPP
