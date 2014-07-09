/**
 * \file ExactExponential.hpp
 * \brief Header for ExactExponential
 *
 * Sample exactly from an exponential distribution.
 *
 * Copyright (c) Charles Karney (2006-2012) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_EXACTEXPONENTIAL_HPP)
#define RANDOMLIB_EXACTEXPONENTIAL_HPP 1

#include <RandomLib/RandomNumber.hpp>

#if defined(_MSC_VER)
// Squelch warnings about constant conditional expressions
#  pragma warning (push)
#  pragma warning (disable: 4127)
#endif

namespace RandomLib {
  /**
   * \brief Sample exactly from an exponential distribution.
   *
   * Sample \e x &ge; 0 from exp(&minus;\e x).  See:
   * - J. von Neumann, Various Techniques used in Connection with Random
   *   Digits, J. Res. Nat. Bur. Stand., Appl. Math. Ser. 12, 36--38
   *   (1951), reprinted in Collected Works, Vol. 5, 768--770 (Pergammon,
   *   1963).
   * - M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions
   *   (National Bureau of Standards, 1964), Sec. 26.8.6.c(2).
   * - G. E. Forsythe, Von Neumann's Comparison Method for Random Sampling from
   *   Normal and Other Distributions, Math. Comp. 26, 817--826 (1972).
   * - D. E. Knuth, TAOCP, Vol 2, Sec 3.4.1.C.3.
   * - D. E. Knuth and A. C. Yao, The Complexity of Nonuniform Random Number
   *   Generation, in "Algorithms and Complexity" (Academic Press, 1976),
   *   pp. 357--428.
   * - P. Flajolet and N. Saheb, The Complexity of Generating an
   *   Exponentially Distributed Variate, J. Algorithms 7, 463--488 (1986).
   *
   * The following code illustrates the basic method given by von Neumann:
   * \code
   // Return a random number x >= 0 distributed with probability exp(-x).
   double ExpDist(RandomLib::Random& r) {
     for (unsigned k = 0; ; ++k) {
       double x = r.Fixed(),    // executed 1/(1-exp(-1)) times on average
         p = x, q;
       do {
         q = r.Fixed();         // executed exp(x)*cosh(x) times on average
         if (!(q < p)) return k + x;
         p = r.Fixed();         // executed exp(x)*sinh(x) times on average
       } while (p < q);
     }
   }
   \endcode
   * This returns a result consuming exp(1)/(1 &minus; exp(-1)) = 4.30 random
   * numbers on average.  (Von Neumann incorrectly states that the method takes
   * (1 + exp(1))/(1 &minus; exp(-1)) = 5.88 random numbers on average.)
   * Because of the finite precision of Random::Fixed(), the code snippet above
   * only approximates exp(&minus;\e x).  Instead, it returns \e x with
   * probability \e h(1 &minus; \e h)<sup><i>x</i>/<i>h</i></sup> for \e x = \e
   * ih, \e h = 2<sup>&minus;53</sup>, and integer \e i &ge; 0.
   *
   * The above is precisely von Neumann's method.  Abramowitz and Stegun
   * consider a variant: sample uniform variants until the first is less than
   * the sum of the rest.  Forsythe converts the < ranking for the runs to &le;
   * which makes the analysis of the discrete case more difficult.  He also
   * drops the "trick" by which the integer part of the deviate is given by the
   * number of rejections when finding the fractional part.
   *
   * Von Neumann says of his method: "The machine has in effect computed a
   * logarithm by performing only discriminations on the relative magnitude of
   * numbers in (0,1).  It is a sad fact of life, however, that under the
   * particular conditions of the Eniac it was slightly quicker to use a
   * truncated power series for log(1&minus;\e T) than to carry out all the
   * discriminations."
   *
   * Here the code is modified to make it more efficient:
   * \code
   // Return a random number x >= 0 distributed with probability exp(-x).
   double ExpDist(RandomLib::Random& r) {
     for (unsigned k = 0; ; ++k) {
       double x = r.Fixed();   // executed 1/(1-exp(-1/2)) times on average
       if (x >= 0.5) continue;
       double p = x, q;
       do {
         q = r.Fixed();        // executed exp(x)*cosh(x) times on average
         if (!(q < p)) return 0.5 * k + x;
         p = r.Fixed();        // executed exp(x)*sinh(x) times on average
       } while (p < q);
     }
   }
   \endcode
   * In addition, the method is extended to use infinite precision uniform
   * deviates implemented by RandomNumber and returning \e exact results for
   * the exponential distribution.  This is possible because only comparisons
   * are done in the method.  The template parameter \e bits specifies the
   * number of bits in the base used for RandomNumber (i.e., base =
   * 2<sup><i>bits</i></sup>).
   *
   * For example the following samples from an exponential distribution and
   * prints various representations of the result.
   * \code
   #include <RandomLib/RandomNumber.hpp>
   #include <RandomLib/ExactExponential.hpp>

     RandomLib::Random r;
     const int bits = 1;
     RandomLib::ExactExponential<bits> edist;
     for (size_t i = 0; i < 10; ++i) {
       RandomLib::RandomNumber<bits> x = edist(r); // Sample
       std::pair<double, double> z = x.Range();
       std::cout << x << " = "     // Print in binary with ellipsis
                 << "(" << z.first << "," << z.second << ")"; // Print range
       double v = x.Value<double>(r); // Round exactly to nearest double
       std::cout << " = " << v << "\n";
     }
   \endcode
   * Here's a possible result: \verbatim
   0.0111... = (0.4375,0.5) = 0.474126
   10.000... = (2,2.125) = 2.05196
   1.00... = (1,1.25) = 1.05766
   0.010... = (0.25,0.375) = 0.318289
   10.1... = (2.5,3) = 2.8732
   0.0... = (0,0.5) = 0.30753
   0.101... = (0.625,0.75) = 0.697654
   0.00... = (0,0.25) = 0.0969214
   0.0... = (0,0.5) = 0.194053
   0.11... = (0.75,1) = 0.867946 \endverbatim
   * First number is in binary with ... indicating an infinite sequence of
   * random bits.  Second number gives the corresponding interval.  Third
   * number is the result of filling in the missing bits and rounding exactly
   * to the nearest representable double.
   *
   * This class uses some mutable RandomNumber objects.  So a single
   * ExactExponential object cannot safely be used by multiple threads.  In a
   * multi-processing environment, each thread should use a thread-specific
   * ExactExponential object.  In addition, these should be invoked with
   * thread-specific random generator objects.
   *
   * @tparam bits the number of bits in each digit.
   **********************************************************************/
  template<int bits = 1> class ExactExponential {
  public:
    /**
     * Return a random deviate with an exponential distribution, exp(&minus;\e
     * x) for \e x &ge; 0.  Requires 7.232 bits per invocation for \e bits = 1.
     * The average number of bits in the fraction = 1.743.  The relative
     * frequency of the results for the fractional part with \e bits = 1 is
     * shown by the histogram
     * \image html exphist.png
     * The base of each rectangle gives the range represented by the
     * corresponding binary number and the area is proportional to its
     * frequency.  A PDF version of this figure is given
     * <a href="exphist.pdf">here</a>.  This allows the figure to be magnified
     * to show the rectangles for all binary numbers up to 9 bits.  Note that
     * this histogram was generated using an earlier version of
     * ExactExponential (thru version 1.3) that implements von Neumann's
     * original method.  The histogram generated with the current version of
     * ExactExponential is the same as this figure for \e u in [0, 1/2].  The
     * histogram for \e u in [1/2, 1] is obtained by shifting and scaling the
     * part for \e u in [0, 1/2] to fit under the exponential curve.
     *
     * Another way of assessing the efficiency of the algorithm is thru the
     * mean value of the balance = (number of random bits consumed) &minus;
     * (number of bits in the result).  If we code the result in mixed Knuth
     * and Yao's unary-binary notation (integer is given in unary, followed by
     * "0" as a separator, followed by the fraction in binary), then the mean
     * balance is 3.906.  (Flajolet and Saheb analyzed the algorithm based on
     * the original von Neumann method and showed that the balance is 5.680 in
     * that case.)
     *
     * For \e bits large, the mean number of random digits is exp(1/2)/(1
     * &minus; exp(&minus;1/2)) = 4.19 (versus 4.30 digits for the original
     * method).
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @return the random sample.
     **********************************************************************/
    template<class Random> RandomNumber<bits> operator()(Random& r) const;
  private:
    /**
     * Return true with probability exp(&minus;\e p) for \e p in (0,1/2).
     **********************************************************************/
    template<class Random> bool
    ExpFraction(Random& r, RandomNumber<bits>& p) const;
    mutable RandomNumber<bits> _x;
    mutable RandomNumber<bits> _v;
    mutable RandomNumber<bits> _w;
  };

  template<int bits> template<class Random> RandomNumber<bits>
  ExactExponential<bits>::operator()(Random& r) const {
    // A simple rejection method gives the 1/2 fractional part.  The number of
    // rejections gives the multiples of 1/2.
    //
    //           bits: used    fract   un-bin  balance double
    // original stats: 9.31615 2.05429 3.63628 5.67987 61.59456
    // new      stats: 7.23226 1.74305 3.32500 3.90725 59.82198
    //
    // The difference between un-bin and fract is exp(1)/(exp(1)-1) = 1.58198
    _x.Init();
    int k = 0;
    while (!ExpFraction(r, _x)) { // Executed 1/(1 - exp(-1/2)) on average
      ++k;
      _x.Init();
    }
    if (k & 1) _x.RawDigit(0) += 1U << (bits - 1);
    _x.AddInteger(k >> 1);
    return _x;
  }

  template<int bits> template<class Random> bool
  ExactExponential<bits>::ExpFraction(Random& r, RandomNumber<bits>& p)
    const {
    // The early bale out
    if (p.Digit(r, 0) >> (bits - 1)) return false;
    // Implement the von Neumann algorithm
    _w.Init();
    if (!_w.LessThan(r, p))     // if (w < p)
      return true;
    while (true) {              // Unroll loop to avoid copying RandomNumber
      _v.Init();                // v = r.Fixed();
      if (!_v.LessThan(r, _w))  // if (v < w)
        return false;
      _w.Init();                // w = r.Fixed();
      if (!_w.LessThan(r, _v))  // if (w < v)
        return true;
    }
  }

} // namespace RandomLib

#if defined(_MSC_VER)
#  pragma warning (pop)
#endif

#endif  // RANDOMLIB_EXACTEXPONENTIAL_HPP
