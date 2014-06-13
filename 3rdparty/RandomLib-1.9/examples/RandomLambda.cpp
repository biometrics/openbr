/**
 * \file RandomLambda.cpp
 * \brief Using the STL and lambda expressions with %RandomLib
 *
 * Compile/link with, e.g.,\n
 * g++ -I../include -O2 -funroll-loops
 *   -o RandomLambda RandomLambda.cpp ../src/Random.cpp\n
 * ./RandomLambda
 *
 * See \ref stl, for more information.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#include <iostream>
#include <vector>
#include <RandomLib/Random.hpp>
#include <RandomLib/NormalDistribution.hpp>

#if (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GNUC_MINOR__ >= 5) || \
    (defined(_MSC_VER) && _MSC_VER >= 1600)
#  define HAVE_LAMBDA 1
#if defined(_MSC_VER)
// Squelch warnings about nonstandard extensions
#  pragma warning (disable: 4239)
#endif
#else
#  define HAVE_LAMBDA 0
#endif

/// \cond SKIP
#if !HAVE_LAMBDA
// Don't need to define these if we can use lambda expressions
template<typename RealType = double> class RandomNormal {
private:
  RandomLib::Random& _r;
  const RandomLib::NormalDistribution<RealType> _n;
  const RealType _mean, _sigma;
  RandomNormal& operator=(const RandomNormal&);
public:
  RandomNormal(RandomLib::Random& r,
               RealType mean = RealType(0), RealType sigma = RealType(1))
    : _r(r), _n(RandomLib::NormalDistribution<RealType>())
    , _mean(mean), _sigma(sigma) {}
  RealType operator()() { return _n(_r, _mean, _sigma); }
};

template<typename IntType = unsigned> class RandomInt {
private:
  RandomLib::Random& _r;
  RandomInt& operator=(const RandomInt&);
public:
  RandomInt(RandomLib::Random& r)
    : _r(r) {}
  IntType operator()(IntType x) { return _r.Integer<IntType>(x); }
};
#endif
/// \endcond

int main(int, char**) {
  RandomLib::Random r; r.Reseed();
#if HAVE_LAMBDA
  std::cout << "Illustrate calling STL routines with lambda expressions\n";
#else
  std::cout << "Illustrate calling STL routines without lambda expressions\n";
#endif
  std::cout << "Using " << r.Name() << "\n"
            << "with seed " << r.SeedString() << "\n\n";

  std::vector<unsigned> c(10);  // Fill with unsigned in [0, 2^32)
#if HAVE_LAMBDA
  std::generate(c.begin(), c.end(),
                [&r]() throw() -> unsigned { return r(); });
#else
  std::generate<std::vector<unsigned>::iterator, RandomLib::Random&>
    (c.begin(), c.end(), r);
#endif

  std::vector<double> b(10);    // Fill with normal deviates
#if HAVE_LAMBDA
  RandomLib::NormalDistribution<> nf;
  std::generate(b.begin(), b.end(),
                [&r, &nf]() throw() -> double
                { return nf(r,0.0,2.0); });
#else
  std::generate(b.begin(), b.end(), RandomNormal<>(r,0.0,2.0));
#endif

  std::vector<int> a(20);  // How to shuffle large vectors
#if HAVE_LAMBDA
  int i = 0;
  std::generate(a.begin(), a.end(),
                [&i]() throw() -> int { return i++; });
  std::random_shuffle(a.begin(), a.end(),
                      [&r](unsigned long long n) throw() -> unsigned long long
                      { return r.Integer<unsigned long long>(n); });
#else
  for (size_t i = 0; i < a.size(); ++i) a[i] = int(i);
  RandomInt<unsigned long long> shuffler(r);
  std::random_shuffle(a.begin(), a.end(), shuffler);
#endif

  return 0;
}
