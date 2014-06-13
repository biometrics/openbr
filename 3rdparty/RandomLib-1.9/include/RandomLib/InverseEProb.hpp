/**
 * \file InverseEProb.hpp
 * \brief Header for InverseEProb
 *
 * Return true with probabililty 1/\e e.
 *
 * Copyright (c) Charles Karney (2012) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_INVERSEEPROB_HPP)
#define RANDOMLIB_INVERSEEPROB_HPP 1

#include <vector>
#include <RandomLib/Random.hpp>

namespace RandomLib {

  /**
   * \brief Return true with probability 1/\e e = exp(&minus;1).
   *
   * InverseEProb p; p(Random& r) returns true with prob 1/\e e using von
   * Neumann's rejection method.  It consumes 4.572 bits per call on average.
   *
   * This class illustrates how to return an exact result using coin tosses
   * only.  A more efficient way of returning an exact result would be to use
   * ExponentialProb p; p(r, 1.0f);
   **********************************************************************/
  class InverseEProb {
  private:
    mutable std::vector<bool> _p;
    template<class Random> bool exph(Random& r) {
      // Return true with prob 1/sqrt(e).
      if (r.Boolean()) return true;
      _p.clear();                      // vector of bits in p
      _p.push_back(false);
      for (bool s = false; ; s = !s) { // s is a parity
        for (size_t i = 0; ; ++i) {    // Compare bits of p and q
          if (i == _p.size())
            _p.push_back(r.Boolean()); // Generate next bit of p if necessary
          if (r.Boolean()) {           // Half the time the bits differ
            if (_p[i]) {        // p's bit is 1, so q is smaller, update p
              _p[i] = false;    // Last bit of q 0
              if (++i < _p.size()) _p.resize(i); // p = q
              break;
            } else
              return s;         // p's bit is 0, so q is bigger, return parity
          } // The other half of the time the bits match, so go to next bit
        }
      }
    }
  public:
    /**
     * Return true with probability 1/\e e.
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @return true with probability 1/\e e.
     **********************************************************************/
    template<class Random> bool operator()(Random& r)
    { return exph(r) && exph(r); }
  };

} // namespace RandomLib

#endif  // RANDOMLIB_INVERSEEPROB_HPP
