/**
 * \file NormalDistribution.hpp
 * \brief Header for NormalDistribution
 *
 * Compute normal deviates.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_NORMALDISTRIBUTION_HPP)
#define RANDOMLIB_NORMALDISTRIBUTION_HPP 1

#include <cmath>                // for std::log

namespace RandomLib {
  /**
   * \brief Normal deviates
   *
   * Sample from the normal distribution.
   *
   * This uses the ratio method; see Knuth, TAOCP, Vol 2, Sec. 3.4.1.C,
   * Algorithm R.  Unlike the Box-Muller method which generates two normal
   * deviates at a time, this method generates just one.  This means that this
   * class has no state that needs to be saved when checkpointing a
   * calculation.  Original citation is\n A. J. Kinderman, J. F. Monahan,\n
   * Computer Generation of Random Variables Using the Ratio of Uniform
   * Deviates,\n ACM TOMS 3, 257--260 (1977).
   *
   * Improved "quadratic" bounds are given by\n J. L. Leva,\n A Fast Normal
   * Random Number Generator,\n ACM TOMS 18, 449--453 and 454--455
   * (1992).
   *
   * The log is evaluated 1.369 times per normal deviate with no bounds, 0.232
   * times with Knuth's bounds, and 0.012 times with the quadratic bounds.
   * Time is approx 0.3 us per deviate (1GHz machine, optimized, RealType =
   * float).
   *
   * Example
   * \code
   *   #include <RandomLib/NormalDistribution.hpp>
   *
   *   RandomLib::Random r;
   *   std::cout << "Seed set to " << r.SeedString() << "\n";
   *   RandomLib::NormalDistribution<double> normdist;
   *   std::cout << "Select from normal distribution:";
   *   for (size_t i = 0; i < 10; ++i)
   *       std::cout << " " << normdist(r);
   *   std::cout << "\n";
   * \endcode
   *
   * @tparam RealType the real type of the results (default double).
   **********************************************************************/
  template<typename RealType = double> class NormalDistribution {
  public:
    /**
     * The type returned by NormalDistribution::operator()(Random&)
     **********************************************************************/
    typedef RealType result_type;
    /**
     * Return a sample of type RealType from the normal distribution with mean
     * &mu; and standard deviation &sigma;.
     *
     * For &mu; = 0 and &sigma; = 1 (the defaults), the distribution is
     * symmetric about zero and is nonzero.  The maximum result is less than 2
     * sqrt(log(2) \e p) where \e p is the precision of real type RealType.
     * The minimum positive value is approximately 1/2<sup><i>p</i>+1</sup>.
     * Here \e p is the precision of real type RealType.
     *
     * @tparam Random the type of RandomCanonical generator.
     * @param[in,out] r the RandomCanonical generator.
     * @param[in] mu the mean value of the normal distribution (default 0).
     * @param[in] sigma the standard deviation of the normal distribution
     *   (default 1).
     * @return the random sample.
     **********************************************************************/
    template<class Random>
    RealType operator()(Random& r, RealType mu = RealType(0),
                        RealType sigma = RealType(1)) const throw();
  };

  template<typename RealType> template<class Random> inline RealType
  NormalDistribution<RealType>::operator()(Random& r, RealType mu,
                                           RealType sigma) const throw() {
    // N.B. These constants can be regarded as "exact", so that the same number
    // of significant figures are used in all versions.  (They serve to
    // "bracket" the real boundary specified by the log expression.)
    const RealType
      m =  RealType( 1.7156  ), // sqrt(8/e) (rounded up)
      s  = RealType( 0.449871), // Constants from Leva
      t  = RealType(-0.386595),
      a  = RealType( 0.19600 ),
      b  = RealType( 0.25472 ),
      r1 = RealType( 0.27597 ),
      r2 = RealType( 0.27846 );
    RealType u, v, Q;
    do {                        // This loop is executed 1.369 times on average
      // Pick point P = (u, v)
      u =  r.template FixedU<RealType>();    // Sample u in (0,1]
      v = m * r.template FixedS<RealType>(); // Sample v in (-m/2, m/2); avoid 0
      // Compute quadratic form Q
      const RealType x = u - s;
      const RealType y = (v < 0 ? -v : v) - t; // Sun has no long double abs!
      Q = x*x + y * (a*y - b*x);
    } while ( Q >= r1 &&        // accept P if Q < r1
              ( Q > r2 ||       // reject P if Q > r2
                v*v > - 4 * u*u * std::log(u) ) ); // accept P if v^2 <= ...
    return mu + sigma * (v / u); // return the slope of P (note u != 0)
  }

} // namespace RandomLib

#endif // RANDOMLIB_NORMALDISTRIBUTION_HPP
