/**
 * \file RandomExample.cpp
 * \brief Simple examples of use of %RandomLib
 *
 * Compile/link with, e.g.,\n
 * g++ -I../include -O2 -funroll-loops
 *   -o RandomExample RandomExample.cpp ../src/Random.cpp\n
 * ./RandomExample
 *
 * This provides a simple illustration of some of the capabilities of
 * %RandomLib.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <RandomLib/Random.hpp>
#include <RandomLib/NormalDistribution.hpp>
#include <RandomLib/RandomSelect.hpp>

int main(int, char**) {
  RandomLib::Random r;          // Create r
  r.Reseed();                   // and give it a unique seed
  std::cout << "Using " << r.Name() << "\n"
            << "with seed " << r.SeedString() << "\n";
  {
    std::cout << "Estimate pi = ";
    size_t in = 0, num = 10000;
    for (size_t i = 0; i < num; ++i) {
      // x, y are in the interval (-1/2,1/2)
      double x = r.FixedS(), y = r.FixedS();
      if (x * x + y * y < 0.25) ++in; // Inside the circle
    }
    std::cout << (4.0 * in) / num << "\n";
  }
  {
    std::cout << "Tossing a coin 20 times: ";
    for (size_t i = 0; i < 20; ++i) std::cout << (r.Boolean() ? "H" : "T");
    std::cout << "\n";
  }
  std::cout << "Generate 20 random bits: " << r.Bits<20>() << "\n";
  {
    std::cout << "Throwing a pair of dice 15 times:";
    for (size_t i = 0; i < 15; ++i)
      std::cout << " " << r.IntegerC(1,6) + r.IntegerC(1,6);
    std::cout << "\n";
  }
  {
    // Weights for throwing a pair of dice
    unsigned w[] = { 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1 };
    // Initialize selection
    RandomLib::RandomSelect<unsigned> sel(w, w + sizeof(w)/sizeof(unsigned));
    std::cout << "A different way of throwing dice:";
    for (size_t i = 0; i < 15; ++i) std::cout << " " << sel(r);
    std::cout << "\n";
  }
  {
    std::cout << "Draw balls from urn containing 5 red and 5 white balls: ";
    int t = 10, w = 5;
    while (t) std::cout << (r.Prob(w, t--) ? w--, "W" : "R");
    std::cout << "\n";
  }
  {
    std::cout << "Shuffling the letters a..z: ";
    std::string digits = "abcdefghijklmnopqrstuvwxyz";
    std::random_shuffle(digits.begin(), digits.end(), r);
    std::cout << digits << "\n";
  }
  {
    std::cout << "Estimate mean and variance of normal distribution: ";
    double m = 0, s = 0;
    int k = 0;
    RandomLib::NormalDistribution<> n;
    while (k < 10000) {
      double x = n(r), m1 = m + (x - m)/++k;
      s += (x - m) * (x - m1); m = m1;
    }
    std::cout << m << ", " << s/(k - 1) << "\n";
  }
  {
    typedef float real;
    enum { prec = 4 };
    std::cout << "Some low precision reals (1/" << (1<<prec) << "):";
    for (size_t i = 0; i < 5; ++i) std::cout << " " << r.Fixed<real, prec>();
    std::cout << "\n";
  }
  std::cout << "Used " << r.Count() << " random numbers\n";
  try {
    // This throws an error if there's a problem
    RandomLib::MRandom32::SelfTest();
    std::cout << "Self test of " << RandomLib::MRandom32::Name()
              << " passed\n";
    RandomLib::MRandom64::SelfTest();
    std::cout << "Self test of " << RandomLib::MRandom64::Name()
              << " passed\n";
    RandomLib::SRandom32::SelfTest();
    std::cout << "Self test of " << RandomLib::SRandom32::Name()
              << " passed\n";
    RandomLib::SRandom64::SelfTest();
    std::cout << "Self test of " << RandomLib::SRandom64::Name()
              << " passed\n";
  }
  catch (std::out_of_range& e) {
    std::cerr << "Self test FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
