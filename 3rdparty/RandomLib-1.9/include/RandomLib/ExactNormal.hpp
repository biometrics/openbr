/**
 * \file ExactNormal.hpp
 * \brief Header for ExactNormal
 *
 * Sample exactly from a normal distribution.
 *
 * Copyright (c) Charles Karney (2011-2012) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_EXACTNORMAL_HPP)
#define RANDOMLIB_EXACTNORMAL_HPP 1

#include <RandomLib/RandomNumber.hpp>
#include <algorithm>            // for max/min

#if defined(_MSC_VER)
// Squelch warnings about constant conditional expressions
#  pragma warning (push)
#  pragma warning (disable: 4127)
#endif

namespace RandomLib {
  /**
   * \brief Sample exactly from a normal distribution.
   *
   * Sample \e x from exp(&minus;<i>x</i><sup>2</sup>/2) / sqrt(2&pi;).  For
   * background, see:
   * - J. von Neumann, Various Techniques used in Connection with Random
   *   Digits, J. Res. Nat. Bur. Stand., Appl. Math. Ser. 12, 36--38
   *   (1951), reprinted in Collected Works, Vol. 5, 768--770 (Pergammon,
   *   1963).
   * - D. E. Knuth and A. C. Yao, The Complexity of Nonuniform Random Number
   *   Generation, in "Algorithms and Complexity" (Academic Press, 1976),
   *   pp. 357--428.
   * - P. Flajolet and N. Saheb, The Complexity of Generating an Exponentially
   *   Distributed Variate, J. Algorithms 7, 463--488 (1986).
   *
   * The algorithm is given in
   * - C. F. F. Karney, <i>Sampling exactly from the normal distribution</i>,
   *   http://arxiv.org/abs/1303.6257 (Mar. 2013).
   * .
   * In brief, the algorithm is:
   * -# Select an integer \e k &ge; 0 with probability
   *    exp(&minus;<i>k</i>/2) (1&minus;exp(&minus;1/2)).
   * -# Accept with probability
   *    exp(&minus; \e k (\e k &minus; 1) / 2); otherwise, reject and start
   *    over at step 1.
   * -# Sample a random number \e x uniformly from [0,1).
   * -# Accept with probability exp(&minus; \e x (\e x + 2\e k) / 2);
   *    otherwise, reject and start over at step 1.
   * -# Set \e x = \e k + \e x.
   * -# With probability 1/2, negate \e x.
   * -# Return \e x.
   * .
   * It is easy to show that this algorithm returns samples from the normal
   * distribution with zero mean and unit variance.  Futhermore, all these
   * steps can be carried out exactly as follows:
   * - Step 1:
   *  - \e k = 0;
   *  - while (ExpProb(&minus;1/2)) increment \e k by 1.
   * - Step 2:
   *  - \e n = \e k (\e k &minus; 1) / 2;
   *  - while (\e n > 0)
   *    { if (!ExpProb(&minus;1/2)) go to step 1; decrement \e n by 1; }
   * - Step 4:
   *  - repeat \e k + 1 times:
   *    if (!ExpProb(&minus; \e x (\e x + 2\e k) / (2\e k + 2))) go to step 1.
   * .
   * Here, ExpProb(&minus;\e p) returns true with probability exp(&minus;\e p).
   * With \e p = 1/2 (steps 1 and 2), this is implemented with von Neumann's
   * rejection technique:
   * - Generate a sequence of random numbers <i>U</i><sub><i>i</i></sub> and
   *   find the greatest \e n such that 1/2 > <i>U</i><sub>1</sub> >
   *   <i>U</i><sub>2</sub> > . . . > <i>U</i><sub><i>n</i></sub>.  (The
   *   resulting value of \e n may be 0.)
   * - If \e n is even, accept and return true; otherwise (\e n odd), reject
   *   and return false.
   * .
   * For \e p = \e x (\e x + 2\e k) / (2\e k + 2) (step 4), we generalize von
   * Neumann's procedure as follows:
   * - Generate two sequences of random numbers <i>U</i><sub><i>i</i></sub>
   *   and  <i>V</i><sub><i>i</i></sub> and find the greatest \e n such that
   *   both the following conditions hold
   *   - \e x > <i>U</i><sub>1</sub> > <i>U</i><sub>2</sub> > . . . >
   *      <i>U</i><sub><i>n</i></sub>;
   *   - <i>V</i><sub><i>i</i></sub> &lt; (\e x + 2 \e k) / (2 \e k + 2) for
   *     all \e i in [1, \e n].
   *   .
   *   (The resulting value of \e n may be 0.)
   * - If \e n is even, accept (return true); otherwise (\e n odd), reject
   *   (return false).
   * .
   * Here, instead of testing <i>V</i><sub><i>i</i></sub> &lt; (\e x + 2 \e k)
   * / (2 \e k + 2), we carry out the following tests:
   * - return true, with probability 2 \e k / (2 \e k + 2);
   * - return false, with probability 1 / (2 \e k + 2);
   * - otherwise (also with probability 1 / (2 \e k + 2)),
   *   return \e x > <i>V</i><sub><i>i</i></sub>.
   * .
   * The resulting method now entails evaluation of simple fractional
   * probabilities (e.g., 1 / (2 \e k + 2)), or comparing random numbers (e.g.,
   * <i>U</i><sub>1</sub> > <i>U</i><sub>2</sub>).  These may be carried out
   * exactly with a finite mean running time.
   *
   * With \e bits = 1, this consumes 30.1 digits on average and the result has
   * 1.19 digits in the fraction.  It takes about 676 ns to generate a result
   * (1460 ns, including the time to round it to a double).  With bits = 32, it
   * takes 437 ns to generate a result (621 ns, including the time to round it
   * to a double).  In contrast, NormalDistribution takes about 44 ns to
   * generate a double result.
   *
   * Another way of assessing the efficiency of the algorithm is thru the mean
   * value of the balance = (number of random bits consumed) &minus; (number of
   * bits in the result).  If we code the result in Knuth & Yao's unary-binary
   * notation, then the mean balance is 26.6.
   *
   * For example the following samples from a normal exponential distribution
   * and prints various representations of the result.
   * \code
   #include <RandomLib/RandomNumber.hpp>
   #include <RandomLib/ExactNormal.hpp>

     RandomLib::Random r;
     const int bits = 1;
     RandomLib::ExactNormal<bits> ndist;
     for (size_t i = 0; i < 10; ++i) {
       RandomLib::RandomNumber<bits> x = ndist(r); // Sample
       std::pair<double, double> z = x.Range();
       std::cout << x << " = "     // Print in binary with ellipsis
                 << "(" << z.first << "," << z.second << ")"; // Print range
       double v = x.Value<double>(r); // Round exactly to nearest double
       std::cout << " = " << v << "\n";
     }
   \endcode
   * Here's a possible result: \verbatim
   -1.00... = (-1.25,-1) = -1.02142
   -0.... = (-1,0) = -0.319708
   0.... = (0,1) = 0.618735
   -0.0... = (-0.5,0) = -0.396591
   0.0... = (0,0.5) = 0.20362
   0.0... = (0,0.5) = 0.375662
   -1.111... = (-2,-1.875) = -1.88295
   -1.10... = (-1.75,-1.5) = -1.68088
   -0.... = (-1,0) = -0.577547
   -0.... = (-1,0) = -0.890553
   \endverbatim
   * First number is in binary with ... indicating an infinite sequence of
   * random bits.  Second number gives the corresponding interval.  Third
   * number is the result of filling in the missing bits and rounding exactly
   * to the nearest representable double.
   *
   * This class uses some mutable RandomNumber objects.  So a single
   * ExactNormal object cannot safely be used by multiple threads.  In a
   * multi-processing environment, each thread should use a thread-specific
   * ExactNormal object.  In addition, these should be invoked with
   * thread-specific random generator objects.
   *
   * @tparam bits the number of bits in each digit.
   **********************************************************************/
  template<int bits = 1> class ExactNormal {
  public:
    /**
     * Return a random deviate with a normal distribution of mean 0 and
     * variance 1.
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @return the random sample.
     **********************************************************************/
    template<class Random> RandomNumber<bits> operator()(Random& r) const;
  private:
    /**
     * Return true with probability exp(&minus;1/2).  For \e bits = 1, this
     * consumes, on average, \e t = 2.846 random digits.  We have \e t = \e a
     * (1&minus;exp(&minus;1/2)) + \e b exp(&minus;1/2), where \e a is the mean
     * bit count for false result = 3.786 and \e b is the mean bit count for
     * true result = 2.236.
     **********************************************************************/
    template<class Random> bool ExpProbH(Random& r) const;

    /**
     * Return true with probability exp(&minus;<i>n</i>/2).  For \e bits = 1,
     * this consumes, on average, \e t
     * (1&minus;exp(&minus;<i>n</i>/2))/(1&minus;exp(&minus;1/2)) random
     * digits.  A true result uses \e n \e b random digits.  A false result
     * uses \e a + \e b [exp(&minus;1/2)/(1&minus;exp(&minus;1/2)) &minus;
     * <i>n</i> exp(&minus;<i>n</i>/2)/(1&minus;exp(&minus;<i>n</i>/2))] random
     * digits.
     **********************************************************************/
    template<class Random> bool ExpProb(Random& r, unsigned n) const;

    /**
     * Return \e n with probability exp(&minus;<i>n</i>/2)
     * (1&minus;exp(&minus;1/2)).  For \e bits = 1, this consumes \e n \e a +
     * \e b random digits if the result is \e n.  Averaging over \e n this
     * becomes (\e b &minus; (\e b &minus; \e a) exp(&minus;1/2))/(1 &minus;
     * exp(&minus;1/2)) digits.
     **********************************************************************/
    template<class Random> unsigned ExpProbN(Random& r) const;

    /**
     * Return true with probability 1/2.  This is similar to r.Boolean() but
     * forces all the random results to come thru RandomNumber::RandomDigit.
     **********************************************************************/
    template<class Random> static bool Boolean(Random& r) {
      // A more general implementation which deals with the case where the base
      // might be negative is:
      //
      //   const unsigned base = 1u << bits;
      //   unsigned b;
      //   do
      //     b = RandomNumber<bits>::RandomDigit(r);
      //   while (b == (base / 2) * 2);
      //   return b & 1u;
      return RandomNumber<bits>::RandomDigit(r) & 1u;
    }

    /**
     * Implement outcomes for choosing with prob (\e x + 2\e k) / (2\e k + 2);
     * return:
     * - 1 (succeed unconditionally) with prob (2\e k) / (2\e k + 2),
     * - 0 (succeed with probability x) with prob 1 / (2\e k + 2),
     * - &minus;1 (fail unconditionally) with prob 1 / (2\e k + 2).
     * .
     * This simulates \code
     double x = r.Fixed();  // Uniform in [0,1)
     x *= (2 * k + 2);
     return x < 2 * k ? 1 : (x < 2 * k + 1 ? 0 : -1);
     \endcode
     **********************************************************************/
    template<class Random> static int Choose(Random& r, int k) {
      // Limit base to 2^15 to avoid integer overflow
      const int b = bits > 15 ? 15 : bits;
      const unsigned mask = (1u << b) - 1;
      const int m = 2 * k + 2;
      int n1 = m - 2, n2 = m - 1;
      // Evaluate u < n/m where u is a random real number in [0,1).  Write u =
      // (d + u') / 2^b where d is a random integer in [0,2^b) and u' is in
      // [0,1).  Then u < n/m becomes u' < n'/m where n' = 2^b * n - d * m and
      // exit if n' <= 0 (false) or n' >= m (true).
      while (true) {
        int d = (mask & RandomNumber<bits>::RandomDigit(r)) * m;
        n1 = (std::max)((n1 << b) - d, 0);
        if (n1 >= m) return 1;
        n2 = (std::min)((n2 << b) - d, m);
        if (n2 <= 0) return -1;
        if (n1 == 0 && n2 == m) return 0;
      }
    }

    mutable RandomNumber<bits> _x;
    mutable RandomNumber<bits> _p;
    mutable RandomNumber<bits> _q;
  };

  template<int bits> template<class Random>
  bool ExactNormal<bits>::ExpProbH(Random& r) const {
    // Bit counts
    // ExpProbH: 2.846 = 3.786 * (1-exp(-1/2)) + 2.236 * exp(-1/2)
    //            t    =  a    * (1-exp(-1/2)) +  b    * exp(-1/2)
    // t = mean bit count for       result = 2.846
    // a = mean bit count for false result = 3.786
    // b = mean bit count for true  result = 2.236
    //
    // for bits large
    //   t = exp(1/2) = 1.6487
    //   a = exp(1/2)/(2*(1-exp(-1/2))) = 2.0951
    //   b = exp(1/2)/(2*exp(-1/2)) = 1.3591
    //
    // Results for Prob(exp(-1)), omitting first test
    // total = 5.889, false = 5.347, true = 6.826
    //
    // Results for Prob(exp(-1)) using ExpProbH(r) && ExpProbH(r),
    // total = 4.572 = (1 - exp(-1)) * a + (1 + exp(-1/2)) * exp(-1/2) * b
    // false = 4.630 = a + b * exp(-1/2)/(1 + exp(-1/2)),
    // true  = 4.472 = 2 * b
    _p.Init();
    if (_p.Digit(r, 0) >> (bits - 1)) return true;
    while (true) {
      _q.Init(); if (!_q.LessThan(r, _p)) return false;
      _p.Init(); if (!_p.LessThan(r, _q)) return true;
    }
  }

  template<int bits> template<class Random>
  bool ExactNormal<bits>::ExpProb(Random& r, unsigned n) const {
    // Bit counts
    // ExpProb(n): t * (1-exp(-n/2))/(1-exp(-1/2))
    // ExpProb(n) = true: n * b
    // ExpProb(n) = false: a +
    //    b * (exp(-1/2)/(1-exp(-1/2)) - n*exp(-n/2)/(1-exp(-n/2)))
    while (n--) { if (!ExpProbH(r)) return false; }
    return true;
  }

  template<int bits> template<class Random>
  unsigned ExactNormal<bits>::ExpProbN(Random& r) const {
    // Bit counts
    // ExpProbN() = n: n * a + b
    unsigned n = 0;
    while (ExpProbH(r)) ++n;
    return n;
  }

  template<int bits> template<class Random> RandomNumber<bits>
  ExactNormal<bits>::operator()(Random& r) const {
    // With bits = 1,
    // - mean number of bits used = 30.10434
    // - mean number of bits in fraction = 1.18700
    // - mean number of bits in result = 3.55257 (unary-binary)
    // - mean balance = 30.10434 - 3.55257 = 26.55177
    // - mean number of bits to generate a double = 83.33398
    // .
    // Note
    // - unary-binary notation (Knuth + Yao, 1976): write x = n + y, with n =
    //   integer and y in [0,1).  If n >=0, then write (n+1) 1's followed by a
    //   0; otherwise (n < 0), write (-n) 0's followed by a 1.  Write y as a
    //   binary fraction.
    // - (bits in result) - (bits in fraction) = 2 (for encoding overhead for
    //   the integer part) + 0.36557, where 0.36557 = (bits used for integer
    //   part) = sum(k*int(sqrt(2/pi)*exp(-x^2/2), x=k..k+1), k=0..inf)
    // - (bits for double) approx (bits used) - (bits in fraction) + 1 (for
    //   guard bit) + 53.41664 where 53.41664 = (bits in fraction of double) =
    //   sum((52-l)*int(sqrt(2/pi)*exp(-x^2/2), x=2^l,2^(l+1)), l=-inf..inf)
    //   This is approximate because it doesn't account for the minimum
    //   exponent, denormalized numbers, and rounding changing the exponent.
    //
    while (true) {
      // Executed sqrt(2/pi)/(1-exp(-1/2)) = 2.027818889827955 times on
      // average.
      unsigned k = ExpProbN(r); // the integer part of the result.
      if (ExpProb(r, (k - 1) * k)) {
        // Probability that this test succeeds is
        // (1 - exp(-1/2)) * sum(exp(-k/2) * exp(-(k-1)*k/2), k=0..inf))
        //   = (1 - exp(-1/2)) * G = 0.689875359564630
        // where G = sum(exp(-k^2/2, k=0..inf) = 1.75331414402145
        // For k == 0, sample from exp(-x^2/2) for x in [0,1].  This succeeds
        // with probability int(exp(-x^2/2),x=0..1).
        //
        // For general k, substitute x' = x + k in exp(-x'^2/2), and obtain
        // exp(-k^2/2) * exp(-x*(x+2*k)/2).  So sample from exp(-x*(x+2*k)/2).
        // This succeeds with probability int(exp(-x*(x+2*k)/2),x=0..1) =
        // int(exp(-x^2/2),x=k..k+1)*exp(k^2/2) =
        //
        //    0.8556243918921 for k = 0
        //    0.5616593588061 for k = 1
        //    0.3963669350376 for k = 2
        //    0.2974440159655 for k = 3
        //    0.2345104783458 for k = 4
        //    0.1921445042826 for k = 5
        //
        // Returns a result with prob sqrt(pi/2) / G = 0.714825772431666;
        // otherwise another trip through the outer loop is taken.
        _x.Init();
        unsigned s = 1;
        for (unsigned j = 0; j <= k; ++j) { // execute k + 1 times
          bool first;
          for (s = 1, first = true; ; s ^= 1, first = false) {
            // A simpler algorithm is indicated by ALT, results in
            // - mean number of bits used = 29.99968
            // - mean number of bits in fraction = 1.55580
            // - mean number of bits in result = 3.92137 (unary-binary)
            // - mean balance = 29.99968 - 3.92137 = 26.07831
            // - mean number of bits to generate a double = 82.86049
            // .
            // This has a smaller balance (by 0.47 bits).  However the number
            // of bits in the fraction is larger by 0.37
            if (first) {        // ALT: if (false) {
              // This implements the success prob (x + 2*k) / (2*k + 2).
              int y = Choose(r, k);
              if (y < 0) break; // the y test fails
              _q.Init();
              if (y > 0) {      // the y test succeeds just test q < x
                if (!_q.LessThan(r, _x)) break;
              } else {          // the y test is ambiguous
                // Test max(q, p) < x.  List _q before _p since it ends up with
                // slightly more digits generated (and these will be used
                // subsequently).  (_p's digits are immediately thrown away.)
                _p.Init(); if (!_x.GreaterPair(r, _q, _p)) break;
              }
            } else {
              // Split off the failure test for k == 0, i.e., factor the prob
              // x/2 test into the product: 1/2 (here) times x (in assignment
              // of y).
              if (k == 0 && Boolean(r)) break;
              // ALT: _q.Init(); if (!_q.LessThan(r, first ? _x : _p)) break;
              _q.Init(); if (!_q.LessThan(r, _p)) break;
              // succeed with prob k == 0 ? x : (x + 2*k) / (2*k + 2)
              int y = k == 0 ? 0 : Choose(r, k);
              if (y < 0)
                break;
              else if (y == 0) {
                _p.Init(); if (!_p.LessThan(r, _x)) break;
              }
            }
            _p.swap(_q);        // a fast way of doing p = q
          }
          if (s == 0) break;
        }
        if (s != 0) {
          _x.AddInteger(k);
          if (Boolean(r)) _x.Negate(); // half of the numbers are negative
          return _x;
        }
      }
    }
  }

} // namespace RandomLib

#if defined(_MSC_VER)
#  pragma warning (pop)
#endif

#endif  // RANDOMLIB_EXACTNORMAL_HPP
