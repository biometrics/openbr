/**
 * \file InversePiProb.hpp
 * \brief Header for InversePiProb
 *
 * Return true with probabililty 1/&pi;.
 *
 * Copyright (c) Charles Karney (2012) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_INVERSEPIPROB_HPP)
#define RANDOMLIB_INVERSEPIPROB_HPP 1

#include <cstdlib>              // for abs(int)
#include <RandomLib/Random.hpp>

namespace RandomLib {

  /**
   * \brief Return true with probability 1/&pi;.
   *
   * InversePiProb p; p(Random& r) returns true with prob 1/&pi; using the
   * method of Flajolet et al.  It consumes 9.6365 bits per call on average.
   *
   * The method is given in Section 3.3 of
   * - P. Flajolet, M. Pelletier, and M. Soria,<br>
   *   On Buffon Machines and Numbers,<br> Proc. 22nd ACM-SIAM Symposium on
   *   Discrete Algorithms (SODA), Jan. 2011.<br>
   *   http://www.siam.org/proceedings/soda/2011/SODA11_015_flajoletp.pdf <br>
   * .
   * using the identity
   * \f[ \frac 1\pi = \sum_{n=0}^\infty
   *      {{2n}\choose n}^3 \frac{6n+1}{2^{8n+2}} \f]
   *
   * It is based on the expression for 1/&pi; given by Eq. (28) of<br>
   * - S. Ramanujan,<br>
   *   Modular Equations and Approximations to &pi;,<br>
   *   Quart. J. Pure App. Math. 45, 350--372 (1914);<br>
   *   In Collected Papers, edited by G. H. Hardy, P. V. Seshu Aiyar,
   *   B. M. Wilson (Cambridge Univ. Press, 1927; reprinted AMS, 2000).<br>
   *   http://books.google.com/books?id=oSioAM4wORMC&pg=PA36 <br>
   * .
   * \f[\frac4\pi = 1 + \frac74 \biggl(\frac 12 \biggr)^3
   * + \frac{13}{4^2} \biggl(\frac {1\cdot3}{2\cdot4} \biggr)^3
   * + \frac{19}{4^3} \biggl(\frac {1\cdot3\cdot5}{2\cdot4\cdot6} \biggr)^3
   * + \ldots \f]
   *
   * The following is a description of how to carry out the algorithm "by hand"
   * with a real coin, together with a worked example:
   * -# Perform three coin tossing experiments in which you toss a coin until
   *    you get tails, e.g., <tt>HHHHT</tt>; <tt>HHHT</tt>; <tt>HHT</tt>. Let
   *    <i>h</i><sub>1</sub> = 4, <i>h</i><sub>2</sub> = 3,
   *    <i>h</i><sub>3</sub> = 2 be the numbers of heads tossed in each
   *    experiment.
   * -# Compute <i>n</i> = &lfloor;<i>h</i><sub>1</sub>/2&rfloor; +
   *    &lfloor;<i>h</i><sub>2</sub>/2&rfloor; +
   *    mod(&lfloor;(<i>h</i><sub>3</sub> &minus; 1)/3&rfloor;, 2) = 2 + 1 + 0
   *    = 3.  Here is a table of the 3 contributions to <i>n</i>:\verbatim
   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17  h
   0  0  1  1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  floor(h1/2)
   0  0  1  1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  floor(h2/2)
   1  0  0  0  1  1  1  0  0  0  1  1  1  0  0  0  1  1  mod(floor((h3-1)/3), 2)
   \endverbatim
   * -# Perform three additional coin tossing experiments in each of which you
   *    toss a coin 2<i>n</i> = 6 times, e.g., <tt>TTHHTH</tt>;
   *    <tt>HHTHH|H</tt>; <tt>THHHHH</tt>.  Are the number of heads and tails
   *    equal in each experiment? <b>yes</b> and <b>no</b> and <b>no</b> &rarr;
   *    <b>false</b>.  (Here, you can give up at the |.)
   * .
   * The final result in this example is <b>false</b>.  The most common way a
   * <b>true</b> result is obtained is with <i>n</i> = 0, in which case the
   * last step vacuously returns <b>true</b>.
   *
   * Proof of the algorithm: Flajolet et al. rearrange Ramanujan's identity as
   * \f[ \frac 1\pi = \sum_{n=0}^\infty
   *      \biggl[{2n\choose n} \frac1{2^{2n}} \biggr]^3
   *      \frac{6n+1}{2^{2n+2}}. \f]
   * Noticing that
   * \f[ \sum_{n=0}^\infty
   *       \frac{6n+1}{2^{2n+2}} = 1, \f]
   * the algorithm becomes:
   * -# pick <i>n</i> &ge; 0 with prob (6<i>n</i>+1) / 2<sup>2<i>n</i>+2</sup>
   *    (mean <i>n</i> = 11/9);
   * -# return <b>true</b> with prob (binomial(2<i>n</i>, <i>n</i>) /
   *    2<sup>2<i>n</i></sup>)<sup>3</sup>.
   *
   * Implement (1) as
   * - geom4(r) + geom4(r) returns <i>n</i> with probability 9(<i>n</i> +
   *   1) / 2<sup>2<i>n</i>+4</sup>;
   * - geom4(r) + geom4(r) + 1 returns <i>n</i> with probability
   *   36<i>n</i> / 2<sup>2<i>n</i>+4</sup>;
   * - combine these with probabilities [4/9, 5/9] to yield (6<i>n</i> +
   *   1) / 2<sup>2<i>n</i>+2</sup>, as required.
   * .
   * Implement (2) as the outcome of 3 coin tossing experiments of 2<i>n</i>
   * tosses with success defined as equal numbers of heads and tails in each
   * trial.
   *
   * This class illustrates how to return an exact result using coin tosses
   * only.  A more efficient implementation (which is still exact) would
   * replace prob59 by r.Prob(5,9) and geom4 by LeadingZeros z; z(r)/2.
   **********************************************************************/
  class InversePiProb {
  private:
    template<class Random> bool prob59(Random& r) {
      // true with prob 5/9 = 0.1 000 111 000 111 000 111 ... (binary expansion)
      if (r.Boolean()) return true;
      for (bool res = false; ; res = !res)
        for (int i = 3; i--; ) if (r.Boolean()) return res;
    }

    template<class Random> int geom4(Random& r) { // Geom(1/4)
      int sum = 0;
      while (r.Boolean() && r.Boolean()) ++sum;
      return sum;
    }

    template<class Random> bool binom(Random& r, int n) {
      // Probability of equal heads and tails on 2*n tosses
      // = binomial(2*n, n) / 2^(2*n)
      int d = 0;
      for (int k = n; k--; ) d += r.Boolean() ? 1 : -1;
      for (int k = n; k--; ) {
        d += r.Boolean() ? 1 : -1;
        // This optimization saves 0.1686 bit per call to operator() on average.
        if (std::abs(d) > k) return false;
      }
      return true;
    }

  public:
    /**
     * Return true with probability 1/&pi;.
     *
     * @tparam Random the type of the random generator.
     * @param[in,out] r a random generator.
     * @return true with probability 1/&pi;.
     **********************************************************************/
    template<class Random> bool operator()(Random& r) {
      // Return true with prob 1/pi.
      int n = geom4(r) + geom4(r) + (prob59(r) ? 1 : 0);
      for (int j = 3; j--; ) if (!binom(r, n)) return false;
      return true;
    }
  };

} // namespace RandomLib

#endif  // RANDOMLIB_INVERSEPIPROB_HPP
