/**
 * \file RandomSeed.hpp
 * \brief Header for RandomSeed
 *
 * This provides a base class for random generators.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_RANDOMSEED_HPP)
#define RANDOMLIB_RANDOMSEED_HPP 1

#include <iostream>
#include <stdexcept>
#include <vector>
#include <iterator>
#include <algorithm>            // For std::transform
#include <sstream>              // For VectorToString
#include <RandomLib/RandomType.hpp>

#if defined(_MSC_VER)
// Squelch warnings about dll vs vector
#pragma warning (push)
#pragma warning (disable: 4251)
#endif

namespace RandomLib {
  /**
   * \brief A base class for random generators
   *
   * This provides facilities for managing the seed and for converting the seed
   * into random generator state.
   *
   * The seed is taken to be a vector of unsigned longs of arbitrary length.
   * (Only the low 32 bit of each element of the vector are used.)  The class
   * provides several methods for setting the seed, static functions for
   * producing "random" and "unique" seeds, and facilities for converting the
   * seed to a string so that it can be printed easily.
   *
   * The seeding algorithms are those used by
   * <a href="http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html">
   * MT19937</a> with some modifications to make all states accessible and to
   * minimize the likelihood of different seeds giving the same state.
   *
   * Finally some low-level routines are provided to facilitate the creation of
   * I/O methods for the random generator.
   *
   * A random generator class can be written based on this class.  The
   * generator class would use the base class methods for setting the seed and
   * for converting the seed into state.  It would provide the machinery for
   * advancing the state and for producing random data.  It is also responsible
   * for the routine to save and restore the generator state (including the
   * seed).
   *
   * Written by Charles Karney <charles@karney.com> and licensed under the
   * MIT/X11 License.  The seeding algorithms are adapted from those of
   * <a href="http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html">
   * MT19937</a>.  For more information, see http://randomlib.sourceforge.net/
   **********************************************************************/
  class RANDOMLIB_EXPORT RandomSeed {
  public:
    typedef Random_u32 u32;
    typedef Random_u64 u64;

    virtual ~RandomSeed() throw() = 0;
    /**
     * A type large enough to hold the seed words.  This is needs to hold 32
     * bits and is an unsigned long for portability.
     **********************************************************************/
    typedef RandomType<32, unsigned long> seed_t;
    typedef seed_t::type seed_type;

    /**
     * \name Resetting the seed
     **********************************************************************/
    ///@{
    /**
     * Set the seed to a vector \e v.  Only the low \e 32 bits of each element
     * are used.
     *
     * @tparam IntType the integral type of the elements of the vector.
     * @param[in] v the vector of elements.
     **********************************************************************/
    template<typename IntType> void Reseed(const std::vector<IntType>& v) {
      Reseed(v.begin(), v.end());
    }
    /**
     * Set the seed to [\e a, \e b) from a pair of iterators.  The iterator
     * must produce results which can be converted into seed_type.  Only the
     * low 32 bits of each element are used.
     *
     * @tparam InputIterator the type of the iterator.
     * @param[in] a the beginning iterator.
     * @param[in] b the ending iterator.
     **********************************************************************/
    template<typename InputIterator>
    void Reseed(InputIterator a, InputIterator b) {
      // Read new seed into temporary so as not to change object on error.
      std::vector<seed_type> t;
      std::transform(a, b, back_inserter(t),
                     seed_t::cast<typename std::iterator_traits<InputIterator>
                     ::value_type>);
      _seed.swap(t);
      Reset();
    }
    /**
     * Set the seed to [\e n].  Only the low 32 bits of \e n are used.
     *
     * @param[in] n the new seed to use.
     **********************************************************************/
    void Reseed(seed_type n) {
      // Reserve space for new seed so as not to change object on error.
      _seed.reserve(1);
      _seed.resize(1);
      _seed[0] = seed_t::cast(n);
      Reset();
    }
    /**
     * Set the seed to [SeedVector()].  This is the standard way to reseed with
     * a "unique" seed.
     **********************************************************************/
    void Reseed() { Reseed(SeedVector()); }
    /**
     * Set the seed from the string \e s using Random::StringToVector.
     *
     * @param[in] s the string to be decoded into a seed.
     **********************************************************************/
    void Reseed(const std::string& s) {
      // Read new seed into temporary so as not to change object on error.
      std::vector<seed_type> t = StringToVector(s);
      _seed.swap(t);
      Reset();
    }
    ///@}

    /**
     * \name Examining the seed
     **********************************************************************/
    ///@{
    /**
     * Return reference to the seed vector (read-only).
     *
     * @return the seed vector.
     **********************************************************************/
    const std::vector<seed_type>& Seed() const throw() { return _seed; }
    /**
     * Format the current seed suitable for printing.
     *
     * @return the seedd as a string.
     **********************************************************************/
    std::string SeedString() const { return VectorToString(_seed); }
    ///@}

    /**
     * \name Resetting the random seed
     **********************************************************************/
    ///@{
    /**
     * Resets the sequence to its just-seeded state.  This needs to be declared
     * virtual here so that the Reseed functions can call it after saving the
     * seed.
     **********************************************************************/
    virtual void Reset() throw() = 0;
    ///@}

    /**
     * \name Static functions for seed management
     **********************************************************************/
    ///@{
    /**
     * Return a 32 bits of data suitable for seeding the random generator.  The
     * result is obtained by combining data from /dev/urandom, gettimeofday,
     * time, and getpid to provide a reasonably "random" word of data.
     * Usually, it is safer to seed the random generator with SeedVector()
     * instead of SeedWord().
     *
     * @return a single "more-or-less random" seed_type to be used as a seed.
     **********************************************************************/
    static seed_type SeedWord();
    /**
     * Return a vector of unsigned longs suitable for seeding the random
     * generator.  The vector is almost certainly unique; however, the results
     * of successive calls to Random::SeedVector() will be correlated.  If
     * several Random objects are required within a single program execution,
     * call Random::SeedVector once, print it out (!), push_back additional
     * data to identify the instance (e.g., loop index, thread ID, etc.), and
     * use the result to seed the Random object.  The number of elements
     * included in the vector may depend on the operating system.  Additional
     * elements may be added in future versions of this library.
     *
     * @return a "unique" vector of seed_type to be uses as a seed.
     **********************************************************************/
    static std::vector<seed_type> SeedVector();
    /**
     * Convert a vector into a string suitable for printing or as an argument
     * for Random::Reseed(const std::string& s).
     *
     * @tparam IntType the integral type of the elements of the vector.
     * @param[in] v the vector to be converted.
     * @return the resulting string.
     **********************************************************************/
    template<typename IntType>
    static std::string VectorToString(const std::vector<IntType>& v) {
      std::ostringstream os;
      os << "[";
      for (typename std::vector<IntType>::const_iterator n = v.begin();
           n != v.end(); ++n) {
        if (n != v.begin())
          os << ",";
        // Normalize in case this is called by user.
        os << seed_t::cast(*n);
      }
      os << "]";
      return os.str();
    }
    /**
     * Convert a string into a vector of seed_type suitable for printing or as
     * an argument for Random::Reseed(const std::vector<seed_type>& v).  Reads
     * consecutive digits in string.  Thus "[1,2,3]" => [1,2,3]; "-0.123e-4" =>
     * [0,123,4], etc.  strtoul understands C's notation for octal and
     * hexadecimal, for example "012 10 0xa" => [10,10,10].  Reading of a
     * number stops at the first illegal character for the base.  Thus
     * "2006-04-08" => [2006,4,0,8] (i.e., 08 becomes two numbers).  Note that
     * input numbers greater than ULONG_MAX overflow to ULONG_MAX, which
     * probably will result in the number being interpreted as LONG_MASK.
     *
     * @param[in] s the string to be converted.
     * @return the resulting vector of seed_type.
     **********************************************************************/
    static std::vector<seed_type> StringToVector(const std::string& s);
    ///@}

  protected:
    /**
     * The seed vector
     **********************************************************************/
    std::vector<seed_type> _seed;

  };

  inline RandomSeed::~RandomSeed() throw() {}

} // namespace RandomLib

#if defined(_MSC_VER)
#pragma warning (pop)
#endif

#endif  // RANDOMLIB_RANDOMSEED_HPP
