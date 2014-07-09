/**
 * \file RandomSave.cpp
 * \brief Examples of saving and restore the state with %RandomLib
 *
 * Compile/link with, e.g.,\n
 * g++ -DHAVE_BOOST_SERIALIZATION -I../include -O2 -funroll-loops
 *   -lboost_serialization-mt -o RandomSave RandomSave.cpp ../src/Random.cpp\n
 * ./RandomSave
 *
 * See \ref save, for more information.
 *
 * Copyright (c) Charles Karney (2011) <charles@karney.com> and licensed under
 * the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#if HAVE_BOOST_SERIALIZATION
#if defined(_MSC_VER)
// Squelch warnings about argument conversion and unchecked parameters with
// boost serialization
#pragma warning (disable: 4244 4996)
#endif
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#endif
#include <RandomLib/Random.hpp>

int main(int, char**) {
  int retval = 0;
  long step = 1000000;
  RandomLib::Random r; r.Reseed();
  std::cout << "Using " << r.Name() << "\n"
            << "with seed " << r.SeedString() << "\n";
  r.StepCount(step);            // use r
  {
    std::cout << "Test save and restore with copy constructor... ";
    RandomLib::Random s(r);     // copy r's state into s via copy constructor
    r.StepCount(step); s.StepCount(step); // step both generators forward
    std::cout << (r == s && r() == s()
                  ? "succeeded\n" : (retval = 1, "failed\n"));
  }
  {
    std::cout << "Test save state via count... ";
    long long savecount = r.Count(); // save the count
    double d = r.Fixed();            // save a random result
    r.StepCount(step);               // step r forward
    r.SetCount(savecount);           // restore to saved count
    std::cout << (d == r.Fixed()
                  ? "succeeded\n" : (retval = 1, "failed\n"));
  }
  {
    std::cout << "Test save state via seed and count... ";
    RandomLib::Random s(r.Seed()); // constructor with r's seed
    s.SetCount(r.Count());         // advance s with r's count
    r.StepCount(step); s.StepCount(step); // step both generators forward
    std::cout << (r == s && r() == s()
                  ? "succeeded\n" : (retval = 1, "failed\n"));
  }
  {
    std::cout << "Test save state to file in portable binary format... ";
    {
      std::ofstream f("rand.bin", std::ios::binary);
      r.Save(f);                // save r in binary mode to rand.bin
    }
    RandomLib::Random s(0);
    {
      std::ifstream f("rand.bin", std::ios::binary);
      s.Load(f);                // load saved state to s
    }
    r.StepCount(step); s.StepCount(step); // step both generators forward
    std::cout << (r == s && r() == s()
                  ? "succeeded\n" : (retval = 1, "failed\n"));
  }
  {
    std::cout << "Test save state to file in text format... ";
    {
      std::ofstream f("rand.txt");
      f << "Random number state:\n" << r << "\n"; // save r with operator<<
    }
    RandomLib::Random s(0);
    {
      std::ifstream f("rand.txt");
      std::string str;
      std::getline(f, str);     // skip over first line
      f >> s;                   // read into s with operator>>
    }
    r.StepCount(step); s.StepCount(step); // step both generators forward
    std::cout << (r == s && r() == s()
                  ? "succeeded\n" : (retval = 1, "failed\n"));
  }
  {
#if HAVE_BOOST_SERIALIZATION
    std::cout << "Test save state to file in boost xml format... ";
    {
      std::ofstream f("rand.xml");
      boost::archive::xml_oarchive oa(f); // set up an xml archive
      oa << BOOST_SERIALIZATION_NVP(r);   // save r to xml file rand.xml
    }
    RandomLib::Random s(0);
    {
      std::ifstream f("rand.xml");
      boost::archive::xml_iarchive ia(f);
      ia >> BOOST_SERIALIZATION_NVP(s); // load saved state to s
    }
    r.StepCount(step); s.StepCount(step); // step both generators forward
    std::cout << (r == s && r() == s()
                  ? "succeeded\n" : (retval = 1, "failed\n"));
#else
    std::cout << "Skipping boost tests\n";
#endif
  }
  return retval;
}
