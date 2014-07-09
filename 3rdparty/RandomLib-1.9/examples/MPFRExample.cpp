/**
 * \file MPFRExample.cpp
 * \brief An example of calling RandomLib::MPFRNormal.
 *
 * Compile, link, and run with, e.g.,
 * g++ -I../include -O2 -o MPFRExample MPFRExample.cpp -lmpfr -lgmp
 * ./MPFRExample
 *
 * Copyright (c) Charles Karney (2012) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#include <cstdio>
#include <iostream>
#include <ctime>                // for time()
#include <RandomLib/MPFRRandom.hpp>
#include <RandomLib/MPFRNormal.hpp>

int main() {
#if HAVE_MPFR
  gmp_randstate_t r;
  gmp_randinit_mt(r);
  time_t t0 = std::time(0);
  gmp_randseed_ui(r, t0);
  mpfr_t z;

  {
    mpfr_prec_t prec = 240; mpfr_init2(z, prec);
    std::cout << "Sample from the unit normal distribution at precision "
              << prec << "\n";
    RandomLib::MPFRNormal<> norm; // bits = 32, by default
    for (int k = 0; k < 10; ++k) {
      norm(z, r, MPFR_RNDN);    // Obtain a normal deviate
      mpfr_out_str(stdout, 10, 0, z, MPFR_RNDN); std::cout << "\n";
    }
  }

  {
    mpfr_prec_t prec = 20; mpfr_set_prec(z, prec);
    std::cout << "Sample ranges from the normal distribution at precision "
              << prec << "\n";
    RandomLib::MPFRNormal<1> norm; // choose bits = 1 so that the ranges
    RandomLib::MPFRRandom<1> x;    // are not too narrow
    for (int k = 0; k < 10; ++k) {
      norm(x, r);               // Obtain an MPFRRandom range
      x(z, MPFR_RNDD);          // Lower bound of range
      std::cout << "[" << mpfr_get_d(z, MPFR_RNDD) << ",";
      x(z, MPFR_RNDU);          // Upper bound of range
      std::cout << mpfr_get_d(z, MPFR_RNDU) << "] -> ";
      x(z, r, MPFR_RNDN);       // Realize the normal deviate
      mpfr_out_str(stdout, 10, 0, z, MPFR_RNDN); std::cout << "\n";
    }
  }

  // Clean up
  mpfr_clear(z); mpfr_free_cache(); gmp_randclear(r);
  return 0;
#else
  std::cerr << "Need MPFR version 3.0 or later to run MPFRExample\n";
  return 1;
#endif  // HAVE_MPFR
}
