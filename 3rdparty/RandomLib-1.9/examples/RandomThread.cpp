/**
 * \file RandomThread.cpp
 * \brief Example of parallelization with %RandomLib using threads
 *
 * Compile/link with, e.g.,\n
 * g++ -I../include -O2 -funroll-loops -fopenmp
 *   -o RandomThread RandomThread.cpp ../src/Random.cpp\n
 * ./RandomThread
 *
 * See \ref parallel, for a description of this example.
 *
 * Copyright (c) Charles Karney (2011) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>

#include <RandomLib/Random.hpp>
#if HAVE_OPENMP
#include <omp.h>
#endif

using RandomLib::Random;

double dotrials(Random& r, int d, long long n) {
  // Require d > 0, n > 0
  double w = 0;
  for ( ; n--; ) {              // Iterate n times
    double h = 0;
    for (int i = 1; i < d; ++i) { // Iterate d-1 times
      double x = 2 * r.FixedS();  // x is in (-1, 1)
      h += x * x;                 // cumulative radius^2
      if (h >= 1) break;          // Point can't be in sphere; bail out,
    }
    // If h < 1 then inside a (d-1) dimensional unit sphere at radius
    // sqrt(h), so extent of last dimension is +/- sqrt(1 -h)
    w += h < 1 ? sqrt(1 - h) : 0;
  }
  return w;
}

double result(int d, long long n, double w) {
  // Volume of (d-1) dimensional box = 2^(d-1).
  // Multiply by another 2 to account for +/- extent in last dimension.
  return double(1U << d) * w / double(n);
}

int usage(const std::string& name, int retval) {
  ( retval == 0 ? std::cout : std::cerr )
    << "Usage: \n" << name
    << " [-d dim] [-n nsamp] [-k ntask] [-l stride] [-s seed] [-h]\n"
    << "Estimate volume of n-dimensional unit sphere\n";
  return retval;
}

int main(int argc, char* argv[]) {
  int d = 4;                    // Number of dimensions
  long long n = 10000000;       // Number of trials 10^7
  int k = 100;                  // Number of tasks
  int l = 4;                    // The leapfrogging stride
  std::string seedstr;
  bool seedgiven = false;

  for (int m = 1; m < argc; ++m) {
    std::string arg(argv[m]);
    if (arg == "-d") {
      if (++m == argc) return usage(argv[0], true);
      std::istringstream str(argv[m]);
      char c;
      if (!(str >> d) || (str >> c) || d <= 0) {
        std::cerr << "Number of dimensions " << argv[m]
                  << " is not a positive number\n";
        return 1;
      }
    } else if (arg == "-n") {
      if (++m == argc) return usage(argv[0], true);
      std::istringstream str(argv[m]);
      char c;
      double fn;
      if (!(str >> fn) || (str >> c) || fn < 1 ||
          fn > double(std::numeric_limits<long long>::max())) {
        std::cerr << "Total count " << argv[m] << " is not a positive number\n";
        return 1;
      }
      n = (long long)(fn);
    } else if (arg == "-k") {
      if (++m == argc) return usage(argv[0], true);
      std::istringstream str(argv[m]);
      char c;
      if (!(str >> k) || (str >> c) || k <= 0) {
        std::cerr << "Number of tasks " << argv[m]
                  << " is not a positive number\n";
        return 1;
      }
    } else if (arg == "-l") {
      if (++m == argc) return usage(argv[0], true);
      std::istringstream str(argv[m]);
      char c;
      if (!(str >> l) || (str >> c) || l <= 0) {
        std::cerr << "Leapfrog stride " << argv[m]
                  << " is not a positive number\n";
        return 1;
      }
    } else if (arg == "-s") {
      seedgiven = true;
      if (++m == argc) return usage(argv[0], 1);
      seedstr = std::string(argv[m]);
    } else
      return usage(argv[0], arg != "-h");
  }

  std::vector<unsigned long> master_seed =
    seedgiven ? Random::StringToVector(seedstr) : Random::SeedVector();

  std::cout << "Estimate volume of a " << d
            << "-dimensional sphere;\nsamples = -n " << n
            << "; tasks = -k " << k
            << "; leapfrog stride = -l " << l << ";\nusing " << Random::Name()
            << "\nwith master seed = -s "
            << Random::VectorToString(master_seed) << ".\n";
  std::cout << "Estimated volume = "
            << std::fixed << std::setprecision(8) << std::flush;

  // Reserve room for a task id at the end of the seed
  master_seed.push_back(0);
  // Fill weight vector with NaNs to verify that all tasks have completed
  std::vector<double> w(k, std::numeric_limits<double>::quiet_NaN());

#if HAVE_OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < k; ++i) { // the main loop over tasks
    Random r;                   // task specific Random
    {
      std::vector<unsigned long> seed(master_seed); // task specific seed
      seed.back() = i / l;      // include task id in seed
      r.Reseed(seed);
      // Turn on leapfrogging with an offset that depends on the task id
      r.SetStride(l, i % l);
    }
    // Do the work; last argument splits n exactly into k pieces
    w[i] = dotrials(r, d, (n * (i + 1))/k - (n * i)/k);
  }
  // Sum up the weights from the individual tasks
  double weight = accumulate(w.begin(), w.end(), 0.0);
  // Compute the result
  std::cout << result(d, n, weight) << "\n";

  return 0;
}
