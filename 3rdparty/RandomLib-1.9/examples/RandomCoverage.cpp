/**
 * \file RandomCoverage.cpp
 * \brief Coverage test for %RandomLib
 *
 * Compile/link with, e.g.,\n
 * g++ -I../include -O2 -funroll-loops
 *   -o RandomCoverage RandomCoverage.cpp ../src/Random.cpp\n
 * ./RandomCoverage
 *
 * This executes nearly all of the public functions in %RandomLib.  This is
 * important, since it allows the compiler to check the code which appears in
 * header files.  It also shows how templated functions can be invoked.
 *
 * Copyright (c) Charles Karney (2006-2012) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <RandomLib/Random.hpp>
#include <RandomLib/NormalDistribution.hpp>
#include <RandomLib/ExponentialDistribution.hpp>
#include <RandomLib/RandomSelect.hpp>
#include <RandomLib/LeadingZeros.hpp>
#include <RandomLib/ExponentialProb.hpp>
#include <RandomLib/RandomNumber.hpp>
#include <RandomLib/UniformInteger.hpp>
#include <RandomLib/ExactExponential.hpp>
#include <RandomLib/ExactNormal.hpp>
#include <RandomLib/DiscreteNormal.hpp>
#include <RandomLib/DiscreteNormalAlt.hpp>
#include <RandomLib/ExactPower.hpp>
#include <RandomLib/InversePiProb.hpp>
#include <RandomLib/InverseEProb.hpp>

#define repeat for (int i = 0; i < 1000; ++i)

void coverage32() {
  typedef RandomLib::SRandom32 Random;
  {
    // Setting and examing the seed + seed management
    std::vector<unsigned long> v(Random::SeedVector());
    unsigned long w = Random::SeedWord();
    std::string s(Random::VectorToString(v));
    { Random r(v); }
    { Random r(v.begin(), v.end()); }
    int a[] = {1, 2, 3, 4};
    { Random r(a, a + 4); }
    { Random r(w); }
    { Random r(s); }
    { Random r; }
    Random r(0);
    r.Reseed(v);
    r.Reseed(v.begin(), v.end());
    r.Reseed(w);
    r.Reseed(s);
    r.Reseed();
    v = r.Seed();
    s = r.SeedString();
    r.Reseed(Random::VectorToString(v));
    r.Reseed(Random::StringToVector(s));
  }
  Random r;
  {
    // Functions returning random integers
    repeat r();
    repeat r.Ran();
    repeat r.Ran32();
    repeat r.Ran64();
    repeat r(52);
    repeat r.Integer<signed char, 3>();
    repeat r.Integer<unsigned char, 3>();
    repeat r.Integer<signed short, 3>();
    repeat r.Integer<3>();
    repeat r.Integer();
    repeat r.Integer<signed short>();
    repeat r.Integer<signed short>(52);
    repeat r.IntegerC<signed short>(51);
    repeat r.IntegerC<signed short>(1,52);
    repeat r.Integer(6u);
    repeat r.IntegerC(5u);
    repeat r.IntegerC(1u,6u);
    repeat r();
    repeat r(52u);
  }
  {
    // Functions returning random reals
    repeat { r.Fixed <float,  16 >(); r.Fixed <float>(); r.Fixed (); }
    repeat { r.FixedU<float,  16 >(); r.FixedU<float>(); r.FixedU(); }
    repeat { r.FixedN<float,  16 >(); r.FixedN<float>(); r.FixedN(); }
    repeat { r.FixedW<float,  16 >(); r.FixedW<float>(); r.FixedW(); }
    repeat { r.FixedS<float,  16 >(); r.FixedS<float>(); r.FixedS(); }
    repeat { r.FixedO<float,  16 >(); r.FixedO<float>(); r.FixedO(); }
    repeat { r.Float <float, 4, 2>(); r.Float <float>(); r.Float (); }
    repeat { r.FloatU<float, 4, 2>(); r.FloatU<float>(); r.FloatU(); }
    repeat { r.FloatN<float, 4, 2>(); r.FloatN<float>(); r.FloatN(); }
    repeat { r.FloatW<float, 4, 2>(); r.FloatW<float>(); r.FloatW(); }
    repeat { r.Real<float>(); r.Real(); }
  }
  {
    // Functions returning other random results
    repeat r.Boolean();
    repeat r.Prob(0.5f);
    repeat r.Prob(2.3, 7.0);
    repeat r.Prob(23, 70);
    repeat r.Bits< 5>();
    repeat r.Bits<64>();
  }
  {
    // Normal distribution
    RandomLib::NormalDistribution<float> nf;
    RandomLib::NormalDistribution<> nd;
    repeat nf(r);
    repeat nd(r, 1.0, 2.0);
  }
  {
    // Exponention distribution
    RandomLib::ExponentialDistribution<float> ef;
    RandomLib::ExponentialDistribution<> ed;
    repeat ef(r);
    repeat ed(r, 2.0);
  }
  {
    // Discrete probabilities
    unsigned w[] = { 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1 };
    std::vector<int> wi(13);
    std::vector<float> wf(13);
    for (int i = 0; i < 13; ++i) {
      wi[i] = int(w[i]);
      wf[i] = float(w[i]);
    }
    { RandomLib::RandomSelect<unsigned> sel; }
    { RandomLib::RandomSelect<unsigned> sel(w, w + 13); }
    { RandomLib::RandomSelect<unsigned> sel(wi.begin(), wi.end()); }
    { RandomLib::RandomSelect<unsigned> sel(wi); }
    { RandomLib::RandomSelect<> sel; }
    { RandomLib::RandomSelect<> sel(w, w + 13); }
    { RandomLib::RandomSelect<> sel(wi); }
    { RandomLib::RandomSelect<> sel(wf.begin(), wf.end()); }
    { RandomLib::RandomSelect<> sel(wf); }
    {
      RandomLib::RandomSelect<unsigned> sel;
      sel.Init(w, w + 13);
      sel.Init(wi.begin(), wi.end());
      sel.Init(wi);
      repeat sel(r);
      sel.TotalWeight();
      sel.MaxWeight();
      sel.Weight(3);
      sel.Choices();
    }
    {
      RandomLib::RandomSelect<> sel;
      sel.Init(w, w + 13);
      sel.Init(wi.begin(), wi.end());
      sel.Init(wi);
      sel.Init(wf);
      repeat sel(r);
      sel.TotalWeight();
      sel.MaxWeight();
      sel.Weight(3);
      sel.Choices();
    }
    // Other distributions
    { RandomLib::LeadingZeros lz; repeat lz(r); }
    { RandomLib::ExponentialProb ep; repeat ep(r, 1.5f); }
    {
      // Infinite precision random numbers
      {
        RandomLib::RandomNumber<1> n;
        n.Init();
        n.Digit(r, 10);
        n.RawDigit(4);
        n.AddInteger(-2);
        n.Negate();
        n.Sign();
        n.Floor();
        n.Ceiling();
        n.Size();
        n.Range();
        n.Fraction<float>(r);
        n.Value<double>(r);
        RandomLib::RandomNumber<1> p;
        n.LessThan(r, p);
        std::ostringstream os;
        os << n;
      }
      {
        RandomLib::RandomNumber<32> n;
        n.Init();
        n.Digit(r, 10);
        n.RawDigit(4);
        n.AddInteger(-2);
        n.Negate();
        n.Sign();
        n.Floor();
        n.Ceiling();
        n.Size();
        n.Range();
        n.Fraction<float>(r);
        n.Value<double>(r);
        RandomLib::RandomNumber<32> p;
        n.LessThan(r, p);
        std::ostringstream os;
        os << n;
      }
      {
        if (RandomLib::UniformInteger<char, 4>::Check(37, 7)) {
          RandomLib::UniformInteger<char, 4> u(r,37);
          u.Min();
          u.Max();
          u.Entropy();
          u.Add(3);
          u.LessThan(r, 4, 1);
          u.LessThanEqual(r, 4, 1);
          u.GreaterThan(r, 60, 7);
          u.GreaterThanEqual(r, 60, 7);
          u.Negate();
          std::ostringstream os;
          os << u;
          os << " " << u(r);
        }
      }
      {
        RandomLib::UniformInteger<short, 4> u(r,37);
        u.Min();
        u.Max();
        u.Entropy();
        u.Add(3);
        u.LessThan(r, 4, 1);
        u.LessThanEqual(r, 4, 1);
        u.GreaterThan(r, 60, 7);
        u.GreaterThanEqual(r, 60, 7);
        u.Negate();
        std::ostringstream os;
        os << u;
        os << " " << u(r);
      }
      {
        RandomLib::UniformInteger<int, 4> u(r,37);
        u.Min();
        u.Max();
        u.Entropy();
        u.Add(3);
        u.LessThan(r, 4, 1);
        u.LessThanEqual(r, 4, 1);
        u.GreaterThan(r, 60, 7);
        u.GreaterThanEqual(r, 60, 7);
        u.Negate();
        std::ostringstream os;
        os << u;
        os << " " << u(r);
      }
      {
        RandomLib::UniformInteger<long long, 32> u(r,37);
        u.Min();
        u.Max();
        u.Entropy();
        u.Add(3);
        u.LessThan(r, 4, 1);
        u.LessThanEqual(r, 4, 1);
        u.GreaterThan(r, 60, 7);
        u.GreaterThanEqual(r, 60, 7);
        u.Negate();
        std::ostringstream os;
        os << u;
        os << " " << u(r);
      }
      // Exact distributions
      { RandomLib::ExactExponential< 1> ed; repeat ed(r).Value<float >(r); }
      { RandomLib::ExactExponential<32> ed; repeat ed(r).Value<double>(r); }
      { RandomLib::ExactNormal< 1> ed; repeat ed(r).Value<float >(r); }
      { RandomLib::ExactNormal<32> ed; repeat ed(r).Value<double>(r); }
      { RandomLib::ExactPower< 1> pd; repeat pd(r,2).Value<float >(r); }
      { RandomLib::ExactPower<32> pd; repeat pd(r,3).Value<double>(r); }
      { RandomLib::DiscreteNormal<short> pd(7); repeat pd(r); }
      { RandomLib::DiscreteNormal<int> pd(7,1,1,2); repeat pd(r); }
      { RandomLib::DiscreteNormal<long> pd(7,1,1,2); repeat pd(r); }
      { RandomLib::DiscreteNormal<long long> pd(7); repeat pd(r); }
      { RandomLib::DiscreteNormalAlt<short,1> pd(7); repeat pd(r)(r); }
      { RandomLib::DiscreteNormalAlt<int,8> pd(7); repeat pd(r)(r); }
      { RandomLib::DiscreteNormalAlt<long,28> pd(7); repeat pd(r)(r); }
      { RandomLib::DiscreteNormalAlt<long long,32> pd(7); repeat pd(r)(r); }
      { RandomLib::InversePiProb pd; repeat pd(r); }
      { RandomLib::InverseEProb pd; repeat pd(r); }
    }
  }
  {
    // Setting position in sequence
    r.Count();
    r.StepCount(1000);
    r.Reset();
    r.SetCount(10000);
    r.SetStride(10,1);
    r.SetStride(5);
    r.GetStride();
    r.SetStride();
  }
  {
    // Other
    Random s(r);
    s = Random::Global;
    void(s == r);
    void(s != s);
    r.swap(s);
    std::swap(r, s);
  }
}

void coverage64() {
  typedef RandomLib::SRandom64 Random;
  {
    // Setting and examing the seed + seed management
    std::vector<unsigned long> v(Random::SeedVector());
    unsigned long w = Random::SeedWord();
    std::string s(Random::VectorToString(v));
    { Random r(v); }
    { Random r(v.begin(), v.end()); }
    int a[] = {1, 2, 3, 4};
    { Random r(a, a + 4); }
    { Random r(w); }
    { Random r(s); }
    { Random r; }
    Random r(0);
    r.Reseed(v);
    r.Reseed(v.begin(), v.end());
    r.Reseed(w);
    r.Reseed(s);
    r.Reseed();
    v = r.Seed();
    s = r.SeedString();
    r.Reseed(Random::VectorToString(v));
    r.Reseed(Random::StringToVector(s));
  }
  Random r;
  {
    // Functions returning random integers
    repeat r();
    repeat r.Ran();
    repeat r.Ran32();
    repeat r.Ran64();
    repeat r(52);
    repeat r.Integer<signed char, 3>();
    repeat r.Integer<unsigned char, 3>();
    repeat r.Integer<signed short, 3>();
    repeat r.Integer<3>();
    repeat r.Integer();
    repeat r.Integer<signed short>();
    repeat r.Integer<signed short>(52);
    repeat r.IntegerC<signed short>(51);
    repeat r.IntegerC<signed short>(1,52);
    repeat r.Integer(6u);
    repeat r.IntegerC(5u);
    repeat r.IntegerC(1u,6u);
    repeat r();
    repeat r(52u);
  }
  {
    // Functions returning random reals
    repeat { r.Fixed <float,  16 >(); r.Fixed <float>(); r.Fixed (); }
    repeat { r.FixedU<float,  16 >(); r.FixedU<float>(); r.FixedU(); }
    repeat { r.FixedN<float,  16 >(); r.FixedN<float>(); r.FixedN(); }
    repeat { r.FixedW<float,  16 >(); r.FixedW<float>(); r.FixedW(); }
    repeat { r.FixedS<float,  16 >(); r.FixedS<float>(); r.FixedS(); }
    repeat { r.FixedO<float,  16 >(); r.FixedO<float>(); r.FixedO(); }
    repeat { r.Float <float, 4, 2>(); r.Float <float>(); r.Float (); }
    repeat { r.FloatU<float, 4, 2>(); r.FloatU<float>(); r.FloatU(); }
    repeat { r.FloatN<float, 4, 2>(); r.FloatN<float>(); r.FloatN(); }
    repeat { r.FloatW<float, 4, 2>(); r.FloatW<float>(); r.FloatW(); }
    repeat { r.Real<float>(); r.Real(); }
  }
  {
    // Functions returning other random results
    repeat r.Boolean();
    repeat r.Prob(0.5f);
    repeat r.Prob(2.3, 7.0);
    repeat r.Prob(23, 70);
    repeat r.Bits< 5>();
    repeat r.Bits<64>();
  }
  {
    // Normal distribution
    RandomLib::NormalDistribution<float> nf;
    RandomLib::NormalDistribution<> nd;
    repeat nf(r);
    repeat nd(r, 1.0, 2.0);
  }
  {
    // Exponention distribution
    RandomLib::ExponentialDistribution<float> ef;
    RandomLib::ExponentialDistribution<> ed;
    repeat ef(r);
    repeat ed(r, 2.0);
  }
  {
    // Discrete probabilities
    unsigned w[] = { 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1 };
    std::vector<int> wi(13);
    std::vector<float> wf(13);
    for (int i = 0; i < 13; ++i) {
      wi[i] = int(w[i]);
      wf[i] = float(w[i]);
    }
    { RandomLib::RandomSelect<unsigned> sel; }
    { RandomLib::RandomSelect<unsigned> sel(w, w + 13); }
    { RandomLib::RandomSelect<unsigned> sel(wi.begin(), wi.end()); }
    { RandomLib::RandomSelect<unsigned> sel(wi); }
    { RandomLib::RandomSelect<> sel; }
    { RandomLib::RandomSelect<> sel(w, w + 13); }
    { RandomLib::RandomSelect<> sel(wi); }
    { RandomLib::RandomSelect<> sel(wf.begin(), wf.end()); }
    { RandomLib::RandomSelect<> sel(wf); }
    {
      RandomLib::RandomSelect<unsigned> sel;
      sel.Init(w, w + 13);
      sel.Init(wi.begin(), wi.end());
      sel.Init(wi);
      repeat sel(r);
      sel.TotalWeight();
      sel.MaxWeight();
      sel.Weight(3);
      sel.Choices();
    }
    {
      RandomLib::RandomSelect<> sel;
      sel.Init(w, w + 13);
      sel.Init(wi.begin(), wi.end());
      sel.Init(wi);
      sel.Init(wf);
      repeat sel(r);
      sel.TotalWeight();
      sel.MaxWeight();
      sel.Weight(3);
      sel.Choices();
    }
    // Other distributions
    { RandomLib::LeadingZeros lz; repeat lz(r); }
    { RandomLib::ExponentialProb ep; repeat ep(r, 1.5f); }
    {
      // Infinite precision random numbers
      {
        RandomLib::RandomNumber<1> n;
        n.Init();
        n.Digit(r, 10);
        n.RawDigit(4);
        n.AddInteger(-2);
        n.Negate();
        n.Sign();
        n.Floor();
        n.Ceiling();
        n.Size();
        n.Range();
        n.Fraction<float>(r);
        n.Value<double>(r);
        RandomLib::RandomNumber<1> p;
        n.LessThan(r, p);
        std::ostringstream os;
        os << n;
      }
      {
        RandomLib::RandomNumber<32> n;
        n.Init();
        n.Digit(r, 10);
        n.RawDigit(4);
        n.AddInteger(-2);
        n.Negate();
        n.Sign();
        n.Floor();
        n.Ceiling();
        n.Size();
        n.Range();
        n.Fraction<float>(r);
        n.Value<double>(r);
        RandomLib::RandomNumber<32> p;
        n.LessThan(r, p);
        std::ostringstream os;
        os << n;
      }
      {
        if (RandomLib::UniformInteger<char, 4>::Check(37, 7)) {
          RandomLib::UniformInteger<char, 4> u(r,37);
          u.Min();
          u.Max();
          u.Entropy();
          u.Add(3);
          u.LessThan(r, 4, 1);
          u.LessThanEqual(r, 4, 1);
          u.GreaterThan(r, 60, 7);
          u.GreaterThanEqual(r, 60, 7);
          u.Negate();
          std::ostringstream os;
          os << u;
          os << " " << u(r);
        }
      }
      {
        RandomLib::UniformInteger<short, 4> u(r,37);
        u.Min();
        u.Max();
        u.Entropy();
        u.Add(3);
        u.LessThan(r, 4, 1);
        u.LessThanEqual(r, 4, 1);
        u.GreaterThan(r, 60, 7);
        u.GreaterThanEqual(r, 60, 7);
        u.Negate();
        std::ostringstream os;
        os << u;
        os << " " << u(r);
      }
      {
        RandomLib::UniformInteger<int, 4> u(r,37);
        u.Min();
        u.Max();
        u.Entropy();
        u.Add(3);
        u.LessThan(r, 4, 1);
        u.LessThanEqual(r, 4, 1);
        u.GreaterThan(r, 60, 7);
        u.GreaterThanEqual(r, 60, 7);
        u.Negate();
        std::ostringstream os;
        os << u;
        os << " " << u(r);
      }
      {
        RandomLib::UniformInteger<long long, 32> u(r,37);
        u.Min();
        u.Max();
        u.Entropy();
        u.Add(3);
        u.LessThan(r, 4, 1);
        u.LessThanEqual(r, 4, 1);
        u.GreaterThan(r, 60, 7);
        u.GreaterThanEqual(r, 60, 7);
        u.Negate();
        std::ostringstream os;
        os << u;
        os << " " << u(r);
      }
      // Exact distributions
      { RandomLib::ExactExponential< 1> ed; repeat ed(r).Value<float >(r); }
      { RandomLib::ExactExponential<32> ed; repeat ed(r).Value<double>(r); }
      { RandomLib::ExactNormal< 1> ed; repeat ed(r).Value<float >(r); }
      { RandomLib::ExactNormal<32> ed; repeat ed(r).Value<double>(r); }
      { RandomLib::ExactPower< 1> pd; repeat pd(r,2).Value<float >(r); }
      { RandomLib::ExactPower<32> pd; repeat pd(r,3).Value<double>(r); }
      { RandomLib::DiscreteNormal<short> pd(7); repeat pd(r); }
      { RandomLib::DiscreteNormal<int> pd(7,1,1,2); repeat pd(r); }
      { RandomLib::DiscreteNormal<long> pd(7,1,1,2); repeat pd(r); }
      { RandomLib::DiscreteNormal<long long> pd(7); repeat pd(r); }
      { RandomLib::DiscreteNormalAlt<short,1> pd(7); repeat pd(r)(r); }
      { RandomLib::DiscreteNormalAlt<int,8> pd(7); repeat pd(r)(r); }
      { RandomLib::DiscreteNormalAlt<long,28> pd(7); repeat pd(r)(r); }
      { RandomLib::DiscreteNormalAlt<long long,32> pd(7); repeat pd(r)(r); }
      { RandomLib::InversePiProb pd; repeat pd(r); }
      { RandomLib::InverseEProb pd; repeat pd(r); }
    }
  }
  {
    // Setting position in sequence
    r.Count();
    r.StepCount(1000);
    r.Reset();
    r.SetCount(10000);
    r.SetStride(10,1);
    r.SetStride(5);
    r.GetStride();
    r.SetStride();
  }
  {
    // Other
    Random s(r);
    s = Random::Global;
    void(s == r);
    void(s != s);
    r.swap(s);
    std::swap(r, s);
  }
}

int main(int, char**) {
  std::cout << "RandomLib Coverage Test\n";
  coverage32();
  coverage64();
  int retval = 0;
  {
    using namespace RandomLib;
    {
      RandomEngine<MT19937<Random_u32>,MixerMT0<Random_u32> >
        s("0x123,0x234,0x345,0x456");
      s.SetCount(999);
      bool pass = s() == 3460025646U;
      std::cout << "Check " << s.Name()
                << (pass ? " passed\n" : " FAILED\n");
      if (!pass) retval = 1;
    }
    {
      RandomEngine<MT19937<Random_u64>,MixerMT0<Random_u64> >
        s("0x12345,0,0x23456,0,0x34567,0,0x45678,0");
      s.SetCount(999);
      bool pass = s() == 994412663058993407ULL;
      std::cout << "Check " << s.Name()
                << (pass ? " passed\n" : " FAILED\n");
      if (!pass) retval = 1;
    }
    {
      SRandomGenerator32 s("0x1234,0x5678,0x9abc,0xdef0");
      s.SetCount(999);
      bool pass = s() == 788493625U;
      std::cout << "Check " << s.Name()
                << (pass ? " passed\n" : " FAILED\n");
      if (!pass) retval = 1;
    }
    {
      SRandomGenerator64 s("5,4,3,2,1");
      s.SetCount(999);
      bool pass = s() == 13356980519185762498ULL;
      std::cout << "Check " << s.Name()
                << (pass ? " passed\n" : " FAILED\n");
      if (!pass) retval = 1;
    }
  }
  return retval;
}
