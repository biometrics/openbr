/**
 * \file LeadingZeros.hpp
 * \brief Header for LeadingZeros
 *
 * Count the leading zeros in a real number.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_LEADINGZEROS_HPP)
#define RANDOMLIB_LEADINGZEROS_HPP 1

namespace RandomLib {
  /**
   * \brief Count of leading zeros.
   *
   * Count of leading zero bits after the binary point in a real number
   * uniformly distributed in (0,1).  (This is equivalent to the geometric
   * distribution with probability 1/2.)  For example
   * \code
   #include <RandomLib/LeadingZeros.hpp>

   RandomLib::Random r; // A RandomGenerator works here too
   std::cout << "Seed set to " << r.SeedString() << "\n";
   LeadingZeros zeros;
   std::cout << "Count of leading zeros:";
   for (size_t i = 0; i < 20; ++i)
       std::cout << " " << zeros(r);
   std::cout << "\n";
   \endcode
   **********************************************************************/
  class LeadingZeros {
  public:
    /**
     * Return the number of zero bits after the binary point in a real number
     * uniformly distributed in (0,1).  Thus \e k is returned with probability
     * 1/2<sup><i>k</i>+1</sup>.  Because MT19937 is \e not a perfect random
     * number generator, this always returns a result in [0, 19937).
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @return the random sample.
     **********************************************************************/
    template<class Random> unsigned operator()(Random& r) const throw();
  };

  template<class Random>
  unsigned LeadingZeros::operator()(Random& r) const throw() {
    // It's simpler to count the number of trailing ones in each w-bit block
    // stopping when we get to a zero bit.
    //
    // Process a word in chunks of size m.  The algorithm here can deal with
    // any m assuming that z is modified accordingly.  m = 4 is an approximate
    // optimum.
    //
    // Can also adapt this routine to use RandomNumber::highest_bit_idx
    // instead.  However the result is considerably slower.
    const int m = 4;
    STATIC_ASSERT(m <= Random::width, "LeadingZeros: m too large");
    // mask with m low bits set
    const typename Random::result_type mask = ~(Random::max << m);
    // Number of trailing 1 bits in [0, 1<<m).  However, correct results are
    // also obtained with any permutation of this array.  This particular
    // permutation is useful since the initial 1/2, 1/4, etc. can be used for
    // m-1, m-2, etc.  To generate the array for the next higher m, append a
    // duplicate of the array and increment the last entry by one.
    const unsigned z[1 << m] =
      { 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, };
    typename Random::result_type x = r();
    for (unsigned b = m, n = 0; b < Random::width; b += m) {
      n += z[x & mask];         // count trailing 1s in chunk
      if (n < b)                // chunk contains a 0
        return n;
      x >>= m;                  // shift out the chunk we've processed
    }
    // x is all ones (prob 1/2^w); process the next word.
    return Random::width + operator()(r);
  }

} // namespace RandomLib

#endif  // RANDOMLIB_LEADINGZEROS_HPP
