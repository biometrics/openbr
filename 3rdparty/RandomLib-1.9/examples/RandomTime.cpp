/**
 * \file RandomTime.cpp
 * \brief Timing %RandomLib
 *
 * Compile/link with, e.g.,\n
 * g++ -I../include -O2 -funroll-loops
 *   -o RandomTime RandomTime.cpp ../src/Random.cpp\n
 * ./RandomTime
 *
 * See \ref timing, for more information.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <ctime>
#if !defined(_MSC_VER)
#include <sys/time.h>
#else
#include <windows.h>
#include <winbase.h>
#endif
#include <RandomLib/Random.hpp>
#include <RandomLib/NormalDistribution.hpp>
#include <RandomLib/RandomSelect.hpp>

#if defined(_MSC_VER)
// Squelch warnings about constant conditional expressions
#  pragma warning (disable: 4127)
#endif

double HighPrecMult() {
#if defined(_MSC_VER)
  LARGE_INTEGER t;
  QueryPerformanceFrequency((LARGE_INTEGER *)&t);
  return 1.0/(t.HighPart*std::pow(2.0, 32) + t.LowPart);
#else
  return 1.e-6;
#endif
}
long long HighPrecTime() {
#if defined(_MSC_VER)
  LARGE_INTEGER t;
  QueryPerformanceCounter((LARGE_INTEGER *)&t);
  return (static_cast<long long>(t.HighPart) << 32) +
    static_cast<long long>(t.LowPart);
#else
  timeval t;
  gettimeofday(&t, NULL);
  return static_cast<long long>(t.tv_sec) * 1000000LL +
    static_cast<long long>(t.tv_usec);
#endif
}

// estime is the estimated time for the command in ns.  The command is executed
// as many time as necessary to fill a second.

#define TIME(expr,esttime) {                                            \
    long long t1, t2;                                                   \
    long long c1 = r.Count();                                           \
    size_t m = int(1.e9/esttime+1);                                     \
    t1=HighPrecTime();                                                  \
    for (size_t j = m; j; --j) { expr; }                                \
    t2=HighPrecTime();                                                  \
    std::cout << std::setprecision(1) << std::setw(8) << std::scientific \
              << 0.1*std::floor((t2-t1)*HighPrecMult()*1.0e10/m+0.5) << "ns "; \
    std::string cmd(#expr);                                             \
    std::string::size_type p;                                           \
    p = cmd.find("template ");                                          \
    if (p != std::string::npos) cmd = cmd.substr(0,p) + cmd.substr(p+9); \
    p = cmd.find(" = ");                                                \
    if (p != std::string::npos) cmd = cmd.substr(p + 3);                \
    p = cmd.find("Random::");                                           \
    if (p != std::string::npos) cmd = cmd.substr(0,p)+cmd.substr(p+8);  \
    p = cmd.find("std::");                                              \
    if (p != std::string::npos) cmd = cmd.substr(0,p)+cmd.substr(p+5);  \
    if (cmd[0] == '(')                                                  \
      cmd = cmd.substr(1,cmd.size()-2);                                 \
    std::cout << std::setprecision(1) << std::setw(5) << std::fixed     \
              << (r.Count()-c1)/float(m) << "rv" << " per " << cmd << "\n"; \
  }

template<typename Random>
void Time(Random& r) {

  volatile bool b = false;
  volatile unsigned i = 0, n = 0;
  volatile typename Random::result_type ii = 0;
  volatile unsigned long long l = 0;
  volatile float f = 0;
  volatile double d = 0;
  std::vector<unsigned long> v;

  ii = r();
  if (ii == 0) n = 1;

  std::cout << "Using " << r.Name() << " with seed "
       << r.SeedString() << "\n";

  std::cout << "Time system random number generator\n";
  TIME(i = rand(),                                   1.0e+01);

  std::cout << "Time generation of integer results\n";
  TIME(ii = r(),                                     2.0e+00);
  TIME(i = r.template Integer<unsigned>(),           2.6e+00);
  TIME(l = r.template Integer<unsigned long long>(), 4.3e+00);
  TIME(i = (r.template Integer<unsigned,6>()),       2.6e+00);
  TIME(i = r.template Integer<unsigned>(52u),        5.6e+00);
  TIME(i = r.template Integer<unsigned>(52u+n),      1.3e+01);

  std::cout << "Time generation of real results\n";
  TIME(f = r.template Fixed<float>(),                4.9e+00);
  TIME(d = r.template Fixed<double>(),               9.5e+00);
  TIME(f = r.template Float<float>(),                1.9e+01);
  TIME(d = r.template Float<double>(),               1.8e+01);

  std::cout << "Time generation of boolean results\n";
  TIME(b = r.template Prob<float>(0.28f),            1.1e+01);
  TIME(b = r.template Prob<double>(0.28),            7.7e+00);

  std::cout << "Time generation of normal distribution\n";
  RandomLib::NormalDistribution<float> nf;
  RandomLib::NormalDistribution<double> nd;
  TIME(f = nf(r),                                    4.1e+01);
  TIME(d = nd(r),                                    5.4e+01);

  std::cout << "Time returning starting seeds\n";
  TIME(i = Random::SeedWord(),                       1.1e+06);
  TIME(v = Random::SeedVector(),                     1.9e+04);

  r.Reset();
  std::cout << "Time getting the ready for first random result\n";
  TIME((r.Reset(), r.SetCount(0)),                   6.9e+03);

  r.SetCount(123);
  std::cout << "Time stepping the generator forward and back\n";
  TIME(r.StepCount(10000),                           6.3e+03);
  TIME(r.StepCount(-10000),                          1.1e+04);

  std::cout << "Time sampling from a discrete distribution\n";
  // Weights for throwing a pair of dice
  unsigned w[] = { 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1 };
  RandomLib::RandomSelect<float> seld(w, w+13);
  TIME(i = seld(r),                                  2.8e+01);
  std::vector<int> a(101);
  for (int m = 0; m < 101; ++m) a[m] = m;

  std::cout << "Time shuffling 100 numbers\n";
  TIME(std::random_shuffle(a.begin(), a.end(), r),   1.3e+03);

  r.SetStride(10);
  std::cout << "Time with stride = 10\n";
  TIME(ii = r(),                                     1.0e+01);
  TIME(d = nd(r),                                    8.7e+01);

  r.SetStride(100);
  std::cout << "Time with stride = 100\n";
  TIME(ii = r(),                                     6.8e+01);
  TIME(d = nd(r),                                    4.1e+02);

  r.SetStride();
  // Avoid warning about set but unused variables
  if (b && i == 0 && l == 0 && f == 0 && d == 0)
    r.StepCount(0);
  return;
}

int main(int, char**) {
  try {
    {
      typedef RandomLib::SRandom32 R;
      R::SelfTest();
      R r;
      r.StepCount(123);
      Time<R>(r);
    }
    {
      typedef RandomLib::SRandom64 R;
      R::SelfTest();
      R r;
      r.StepCount(123);
      Time<R>(r);
    }
    if (false) {
      // Skip timing MRandom{32,64}
      {
        typedef RandomLib::MRandom32 R;
        R::SelfTest();
        R r;
        r.StepCount(123);
        Time<R>(r);
      }
      {
        typedef RandomLib::MRandom64 R;
        R::SelfTest();
        R r;
        r.StepCount(123);
        Time<R>(r);
      }
    }
    return 0;
  }
  catch (const std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << "\n";
    return 1;
  }
  catch (...) {
    std::cerr << "Caught unknown exception\n";
    return 1;
  }
}
