/**
 * \file ExponentialDistribution.hpp
 * \brief Header for ExponentialDistribution
 *
 * Sample from an exponential distribution.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_EXPONENTIALDISTRIBUTION_HPP)
#define RANDOMLIB_EXPONENTIALDISTRIBUTION_HPP 1

#include <cmath>

namespace RandomLib {
  /**
   * \brief The exponential distribution.
   *
   * Sample from the distribution exp(&minus;<i>x</i>/&mu;) for \e x &ge; 0.
   * This uses the logarithm method, see Knuth, TAOCP, Vol 2, Sec 3.4.1.D.
   * Example \code
   #include <RandomLib/ExponentialDistribution.hpp>

     RandomLib::Random r;
     std::cout << "Seed set to " << r.SeedString() << "\n";
     RandomLib::ExponentialDistribution<double> expdist;
     std::cout << "Select from exponential distribution:";
     for (size_t i = 0; i < 10; ++i)
         std::cout << " " << expdist(r);
     std::cout << "\n";
   \endcode
   *
   * @tparam RealType the real type of the results (default double).
   **********************************************************************/
  template<typename RealType = double> class ExponentialDistribution {
  public:
    /**
     * The type returned by ExponentialDistribution::operator()(Random&)
     **********************************************************************/
    typedef RealType result_type;
    /**
     * Return a sample of type RealType from the exponential distribution and
     * mean &mu;.  This uses Random::FloatU() which avoids taking log(0) and
     * allows rare large values to be returned.  If &mu; = 1, minimum returned
     * value = 0 with prob 1/2<sup><i>p</i></sup>; maximum returned value =
     * log(2)(\e p + \e e) with prob 1/2<sup><i>p</i> + <i>e</i></sup>.  Here
     * \e p is the precision of real type RealType and \e e is the exponent
     * range.
     *
     * @tparam Random the type of RandomCanonical generator.
     * @param[in,out] r the RandomCanonical generator.
     * @param[in] mu the mean value of the exponential distribution (default 1).
     * @return the random sample.
     **********************************************************************/
    template<class Random>
    RealType operator()(Random& r, RealType mu = RealType(1)) const throw();
  };

  template<typename RealType>  template<class Random> inline RealType
  ExponentialDistribution<RealType>::operator()(Random& r, RealType mu) const
    throw() {
    return -mu * std::log(r.template FloatU<RealType>());
  }

} // namespace RandomLib

#endif // RANDOMLIB_EXPONENTIALDISTRIBUTION_HPP
