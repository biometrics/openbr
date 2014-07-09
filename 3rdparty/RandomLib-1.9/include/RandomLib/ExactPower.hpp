/**
 * \file ExactPower.hpp
 * \brief Header for ExactPower
 *
 * Sample exactly from a power distribution.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_EXACTPOWER_HPP)
#define RANDOMLIB_EXACTPOWER_HPP 1

#include <RandomLib/RandomNumber.hpp>

namespace RandomLib {
  /**
   * \brief Sample exactly from a power distribution.
   *
   * Sample exactly from power distribution (<i>n</i> + 1)
   * <i>x</i><sup><i>n</i></sup> for \e x in (0,1) and integer \e n &ge; 0 using
   * infinite precision.  The template parameter \e bits specifies the number
   * of bits in the base used for RandomNumber (i.e., base =
   * 2<sup><i>bits</i></sup>).
   *
   * This class uses some mutable RandomNumber objects.  So a single ExactPower
   * object cannot safely be used by multiple threads.  In a multi-processing
   * environment, each thread should use a thread-specific ExactPower object.
   * In addition, these should be invoked with thread-specific random generator
   * objects.
   *
   * @tparam bits the number of bits in each digit.
   **********************************************************************/
  template<int bits = 1> class ExactPower {
  public:
    /**
     * Return the random deviate with a power distribution, (<i>n</i> + 1)
     * <i>x</i><sup><i>n</i></sup> for \e x in (0,1) and integer \e n &ge; 0.
     * Returned result is a RandomNumber with base 2<sup><i>bits</i></sup>.
     * For \e bits = 1, the number of random bits in the result and consumed
     * are as follows: \verbatim
      n    random bits
          result  consumed
      0   0        0
      1   2        4
      2   2.33     6.67
      3   2.67     9.24
      4   2.96    11.71
      5   3.20    14.11
      6   3.41    16.45
      7   3.59    18.75
      8   3.75    21.01
      9   3.89    23.25
     10   4.02    25.47
     \endverbatim
     * The relative frequency of the results with \e bits = 1 and \e n = 2 can
     * be is shown by the histogram
     * \image html powerhist.png
     * The base of each rectangle gives the range represented by the
     * corresponding binary number and the area is proportional to its
     * frequency.  A PDF version of this figure
     * <a href="powerhist.pdf">here</a>.  This allows the figure to be
     * magnified to show the rectangles for all binary numbers up to 9 bits.
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @param[in] n the power.
     * @return the random sample.
     **********************************************************************/
    template<class Random>
    RandomNumber<bits> operator()(Random& r, unsigned n) const;
  private:
    mutable RandomNumber<bits> _x;
    mutable RandomNumber<bits> _y;
  };

  template<int bits> template<class Random> RandomNumber<bits>
  ExactPower<bits>::operator()(Random& r, unsigned n) const {
    // Return max(u_0, u_1, u_2, ..., u_n).  Equivalent to taking the
    // (n+1)th root of u_0.
    _x.Init();
    for (; n--;) {
      _y.Init();
      // For bits = 1, we can save 1 bit on the first iteration by using a
      // technique suggested by Knuth and Yao (1976).  When comparing the
      // digits of x and y, use 1 bit to determine if the digits are the same.
      // If they are, then use another bit for the value of the digit.  If they
      // are not, then the next bit of the maximum must be 1 (avoiding using a
      // second bit).  Applying this optimization to subsequent iterations
      // (when x already is filled with some bits) gets trickier.
      if (_x.LessThan(r, _y))
        _x.swap(_y);            // x = y;
    }
    return _x;
  }

} // namespace RandomLib

#endif  // RANDOMLIB_EXACTPOWER_HPP
