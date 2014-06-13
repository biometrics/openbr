/**
 * \file Random.cpp
 * \brief Implementation code for %RandomLib
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 *
 * \brief Code for MixerMT0, MixerMT1, MixerSFMT.
 *
 * MixerMT0 is adapted from MT19937 (init_by_array) and MT19937_64
 * (init_by_array64) by Makoto Matsumoto and Takuji Nishimura.  See
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c and
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/VERSIONS/C-LANG/mt19937-64.c
 *
 * MixerMT1 contains modifications to MixerMT0 by Charles Karney to
 * correct defects in MixerMT0.  This is described in W. E. Brown,
 * M. Fischler, J. Kowalkowski, M. Paterno, Random Number Generation in C++0X:
 * A Comprehensive Proposal, version 3, Sept 2006, Sec. 26.4.7.1,
 * http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2079.pdf
 * This has been replaced in the C++11 standard by MixerSFMT.
 *
 * MixerSFMT is adapted from SFMT19937's init_by_array Mutsuo Saito given in
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/SFMT-src-1.2.tar.gz and
 * is part of the C++11 standard; see P. Becker, Working Draft, Standard for
 * Programming Language C++, Oct. 2007, Sec. 26.4.7.1,
 * http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2461.pdf
 *
 * The adaption to the C++ is copyright (c) Charles Karney (2006-2011)
 * <charles@karney.com> and licensed under the MIT/X11 License.  For more
 * information, see http://randomlib.sourceforge.net/
 *
 * \brief Code for MT19937<T> and SFMT19937<T>.
 *
 * MT19937<T> is adapted from MT19937 and MT19937_64 by Makoto Matsumoto and
 * Takuji Nishimura.  See
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c and
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/VERSIONS/C-LANG/mt19937-64.c
 *
 * The code for stepping MT19937 backwards is adapted (and simplified) from
 * revrand() by Katsumi Hagita.  See
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/VERSIONS/FORTRAN/REVmt19937b.f
 *
 * SFMT19937<T> is adapted from SFMT19937 Mutsuo Saito given in
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/M062821.pdf and
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/SFMT-src-1.2.tar.gz
 *
 * The code for stepping SFMT19937 backwards is by Charles Karney.
 *
 * The adaption to the C++ is copyright (c) Charles Karney (2006-2011)
 * <charles@karney.com> and licensed under the MIT/X11 License.  For more
 * information, see http://randomlib.sourceforge.net/
 **********************************************************************/

#define RANDOMLIB_RANDOM_CPP 1

/**
 * Let the header file know that the library is being built.
 **********************************************************************/
#define RANDOMLIB_BUILDING_LIBRARY 1

#include <RandomLib/Random.hpp>

#if defined(_MSC_VER) || defined(__MINGW32__)
#define RANDOMLIB_WINDOWS 1
#else
#define RANDOMLIB_WINDOWS 0
#endif

#include <fstream>              // For SeedWord reading /dev/urandom
#include <ctime>                // For SeedWord calling time()
#include <sstream>              // For formatting in Write32/Read32
#include <iomanip>              // For formatting in Write32/Read32
#if !RANDOMLIB_WINDOWS
#include <sys/time.h>           // For SeedWord calling gettimeofday
#include <unistd.h>             // For SeedWord calling getpid(), gethostid()
#else
#include <windows.h>            // For SeedWord calling high prec timer
#include <winbase.h>
#include <process.h>            // For SeedWord calling getpid()
#define getpid _getpid
#define gmtime_r(t,g) gmtime_s(g,t)
#endif

#if RANDOMLIB_WINDOWS || defined(__CYGWIN__)
#define strtoull strtoul
#endif

#if defined(_MSC_VER)
// Squelch warnings about constant conditional expressions
#  pragma warning (disable: 4127)
#endif

namespace RandomLib {

  // RandomType implementation

  template<>
  void Random_u32::Write32(std::ostream& os, bool bin, int& cnt,
                           Random_u32::type x) {
    if (bin) {
      unsigned char buf[4];
      // Use network order -- most significant byte first
      buf[3] = (unsigned char)(x);
      buf[2] = (unsigned char)(x >>= 8);
      buf[1] = (unsigned char)(x >>= 8);
      buf[0] = (unsigned char)(x >>= 8);
      os.write(reinterpret_cast<const char *>(buf), 4);
    } else {
      const int longsperline = 72/9;
      // Use hexadecimal to minimize storage together with stringstream to
      // isolate the effect of changing the base.
      std::ostringstream str;
      // No spacing before or after
      if (cnt > 0)
        // Newline every longsperline longs
        str << (cnt % longsperline ? ' ' : '\n');
      str << std::hex << x;
      os << str.str();
      ++cnt;
    }
  }

  template<>
  void Random_u32::Read32(std::istream& is, bool bin, Random_u32::type& x) {
    if (bin) {
      unsigned char buf[4];
      is.read(reinterpret_cast<char *>(buf), 4);
      // Use network order -- most significant byte first
      x = Random_u32::type(buf[0]) << 24 | Random_u32::type(buf[1]) << 16 |
        Random_u32::type(buf[2]) << 8 | Random_u32::type(buf[3]);
    } else {
      std::string s;
      is >> std::ws >> s;
      // Use hexadecimal to minimize storage together with stringstream to
      // isolate the effect of changing the base.
      std::istringstream str(s);
      str >> std::hex >> x;
    }
    x &= Random_u32::mask;
  }

  template<>
  void Random_u64::Write32(std::ostream& os, bool bin, int& cnt,
                           Random_u64::type x) {
    Random_u32::Write32(os, bin, cnt, Random_u32::cast(x >> 32));
    Random_u32::Write32(os, bin, cnt, Random_u32::cast(x      ));
  }

  template<>
  void Random_u64::Read32(std::istream& is, bool bin, Random_u64::type& x) {
    Random_u32::type t;
    Random_u32::Read32(is, bin, t);
    x = Random_u64::type(t) << 32;
    Random_u32::Read32(is, bin, t);
    x |= Random_u64::type(t);
  }

  // RandomSeed implementation

  RandomSeed::seed_type RandomSeed::SeedWord() {
    // Check that the assumptions made about the capabilities of the number
    // system are valid.
    STATIC_ASSERT(std::numeric_limits<seed_type>::radix == 2 &&
                  !std::numeric_limits<seed_type>::is_signed &&
                  std::numeric_limits<seed_type>::digits >= 32,
                  "seed_type is a bad type");
    u32::type t = 0;
    // Linux has /dev/urandom to initialize the seed randomly.  (Use
    // /dev/urandom instead of /dev/random because it does not block.)
    {
      std::ifstream f("/dev/urandom", std::ios::binary | std::ios::in);
      if (f.good()) {
        // Read 32 bits from /dev/urandom
        f.read(reinterpret_cast<char *>(&t), sizeof(t));
      }
    }
    std::vector<seed_type> v = SeedVector();
    for (size_t i = v.size(); i--;)
      u32::CheckSum(u32::type(v[i]), t);
    return seed_t::cast(t);
  }

  std::vector<RandomSeed::seed_type> RandomSeed::SeedVector() {
    std::vector<seed_type> v;
    {
      // fine-grained timer
#if !RANDOMLIB_WINDOWS
      timeval tv;
      if (gettimeofday(&tv, 0) == 0)
        v.push_back(seed_t::cast(tv.tv_usec));
#else
      LARGE_INTEGER taux;
      if (QueryPerformanceCounter((LARGE_INTEGER *)&taux)) {
        v.push_back(seed_t::cast(taux.LowPart));
        v.push_back(seed_t::cast(taux.HighPart));
      }
#endif
    }
    // seconds
    const time_t tim = std::time(0);
    v.push_back(seed_t::cast(seed_type(tim)));
    // PID
    v.push_back(seed_t::cast(getpid()));
#if !RANDOMLIB_WINDOWS
    // host ID
    v.push_back(seed_t::cast(gethostid()));
#endif
    {
      // year
#if !defined(__MINGW32__)
      tm gt;
      gmtime_r(&tim, &gt);
      v.push_back((seed_type(1900) + seed_t::cast(gt.tm_year)));
#else
      tm* gt = gmtime(&tim);
      v.push_back((seed_type(1900) + seed_t::cast(gt->tm_year)));
#endif
    }
    // Candidates for additional elements:
    // ip address(es) of computer, thread index.
    std::transform(v.begin(), v.end(), v.begin(), seed_t::cast<seed_type>);
    return v;
  }

  std::vector<RandomSeed::seed_type>
  RandomSeed::StringToVector(const std::string& s) {
    std::vector<seed_type> v(0);
    const char* c = s.c_str();
    char* q;
    std::string::size_type p = 0;
    while (true) {
      p = s.find_first_of("0123456789", p);
      if (p == std::string::npos)
        break;
      v.push_back(seed_t::cast(std::strtoull(c + p, &q, 0)));
      p = q - c;
    }
    return v;
  }

  // RandomEngine implementation

  template<class Algorithm, class Mixer>
  void RandomEngine<Algorithm, Mixer>::Init() throw() {
    // On exit we have _ptr == N.

    STATIC_ASSERT(std::numeric_limits<typename mixer_t::type>::radix == 2 &&
                  !std::numeric_limits<typename mixer_t::type>::is_signed &&
                  std::numeric_limits<typename mixer_t::type>::digits >=
                  int(mixer_t::width),
                  "mixer_type is a bad type");

    STATIC_ASSERT(std::numeric_limits<result_type>::radix == 2 &&
                  !std::numeric_limits<result_type>::is_signed &&
                  std::numeric_limits<result_type>::digits >= width,
                  "engine_type is a bad type");

    STATIC_ASSERT(mixer_t::width == 32 || mixer_t::width == 64,
                  "Mixer width must be 32 or 64");

    STATIC_ASSERT(width == 32 || width == 64,
                  "Algorithm width must be 32 or 64");

    // If the bit-widths are the same then the data sizes must be the same.
    STATIC_ASSERT(!(mixer_t::width == width) ||
                  sizeof(_stateu) == sizeof(_state),
                  "Same bit-widths but different storage");

    // Repacking assumes that narrower data type is at least as wasteful than
    // the broader one.
    STATIC_ASSERT(!(mixer_t::width < width) ||
                  sizeof(_stateu) >= sizeof(_state),
                  "Narrow data type uses less storage");

    STATIC_ASSERT(!(mixer_t::width > width) ||
                  sizeof(_stateu) <= sizeof(_state),
                  "Narrow data type uses less storage");

    // Require that _statev and _state are aligned since no repacking is done
    // when calling Transition
    STATIC_ASSERT(sizeof(_statev) == sizeof(_state),
                  "Storage mismatch with internal engine data type");

    // Convert the seed into state
    Mixer::SeedToState(_seed, _stateu, NU);

    // Pack into _state
    if (mixer_t::width < width) {
      for (size_t i = 0; i < N; ++i)
        // Assume 2:1 LSB packing
        _state[i] = result_type(_stateu[2*i]) |
          result_type(_stateu[2*i + 1]) <<
          (mixer_t::width < width ? mixer_t::width : 0);
    } else if (mixer_t::width > width) {
      for (size_t i = N; i--;)
        // Assume 1:2 LSB packing
        _state[i] = result_t::cast(_stateu[i>>1] >> width * (i&1u));
    } // Otherwise the union takes care of it

    Algorithm::NormalizeState(_state);

    _rounds = -1;
    _ptr = N;
  }

  template<class Algorithm, class Mixer> Random_u32::type
  RandomEngine<Algorithm, Mixer>::Check(u64::type v, u32::type e,
                                        u32::type m) const {
    if (v != version)
      throw RandomErr(Name() + ": Unknown version");
    if (e != Algorithm::version)
      throw RandomErr(Name() + ": Algorithm mismatch");
    if (m != Mixer::version)
      throw RandomErr(Name() + ": Mixer mismatch");
    u32::type check = 0;
    u64::CheckSum(v, check);
    u32::CheckSum(e, check);
    u32::CheckSum(m, check);
    u32::CheckSum(u32::type(_seed.size()), check);
    for (std::vector<seed_type>::const_iterator n = _seed.begin();
         n != _seed.end(); ++n) {
      if (*n != seed_t::cast(*n))
        throw RandomErr(Name() + ": Illegal seed value");
      u32::CheckSum(u32::type(*n), check);
    }
    u32::CheckSum(_ptr, check);
    if (_stride == 0 || _stride > UNINIT/2)
      throw RandomErr(Name() + ": Invalid stride");
    u32::CheckSum(_stride, check);
    if (_ptr != UNINIT) {
      if (_ptr >= N + _stride)
        throw RandomErr(Name() + ": Invalid pointer");
      u64::CheckSum(_rounds, check);
      Algorithm::CheckState(_state, check);
    }
    return check;
  }

  template<typename Algorithm, typename Mixer>
  RandomEngine<Algorithm, Mixer>::RandomEngine(std::istream& is, bool bin) {
    u64::type versionr;
    u32::type versione, versionm, t;
    u64::Read32(is, bin, versionr);
    u32::Read32(is, bin, versione);
    u32::Read32(is, bin, versionm);
    u32::Read32(is, bin, t);
    _seed.resize(size_t(t));
    for (std::vector<seed_type>::iterator n = _seed.begin();
         n != _seed.end(); ++n) {
      u32::Read32(is, bin, t);
      *n = seed_type(t);
    }
    u32::Read32(is, bin, t);
    // Don't need to worry about sign extension because _ptr is unsigned.
    _ptr = unsigned(t);
    u32::Read32(is, bin, t);
    _stride = unsigned(t);
    if (_ptr != UNINIT) {
      u64::type p;
      u64::Read32(is, bin, p);
      _rounds = (long long)(p);
      // Sign extension in case long long is bigger than 64 bits.
      _rounds <<= 63 - std::numeric_limits<long long>::digits;
      _rounds >>= 63 - std::numeric_limits<long long>::digits;
      for (unsigned i = 0; i < N; ++i)
        result_t::Read32(is, bin, _state[i]);
    }
    u32::Read32(is, bin, t);
    if (t != Check(versionr, versione, versionm))
      throw RandomErr(Name() + ": Checksum failure");
  }

  template<typename Algorithm, typename Mixer>
  void RandomEngine<Algorithm, Mixer>::Save(std::ostream& os,
                                            bool bin) const {
    u32::type check = Check(version, Algorithm::version, Mixer::version);
    int c = 0;
    u64::Write32(os, bin, c, version);
    u32::Write32(os, bin, c, Algorithm::version);
    u32::Write32(os, bin, c, Mixer::version);
    u32::Write32(os, bin, c, u32::type(_seed.size()));
    for (std::vector<seed_type>::const_iterator n = _seed.begin();
         n != _seed.end(); ++n)
      u32::Write32(os, bin, c, u32::type(*n));
    u32::Write32(os, bin, c, _ptr);
    u32::Write32(os, bin, c, _stride);
    if (_ptr != UNINIT) {
      u64::Write32(os, bin, c, u64::type(_rounds));
      for (unsigned i = 0; i < N; ++i)
        result_t::Write32(os, bin, c, _state[i]);
    }
    u32::Write32(os, bin, c, check);
  }

  template<typename Algorithm, typename Mixer>
  void RandomEngine<Algorithm, Mixer>::StepCount(long long n) throw() {
    // On exit we have 0 <= _ptr <= N.
    if (_ptr == UNINIT)
      Init();
    const long long ncount = n + Count(); // new Count()
    long long nrounds = ncount / N;
    int nptr = int(ncount - nrounds * N);
    // We pick _ptr = N or _ptr = 0 depending on which choice involves the
    // least work.  We thus avoid doing one (potentially unneeded) call to
    // Transition.
    if (nptr < 0) {
      --nrounds;
      nptr += N;
    } else if (nptr == 0 && nrounds > _rounds) {
      nptr = N;
      --nrounds;
    }
    if (nrounds != _rounds)
      Algorithm::Transition(nrounds - _rounds, _statev);
    _rounds = nrounds;
    _ptr = nptr;
  }

  template<typename Algorithm, typename Mixer>
  void RandomEngine<Algorithm, Mixer>::SelfTest() {
    RandomEngine g(std::vector<seed_type>(0));
    g.SetCount(10000-1);
    result_type x = g();
    if (SelfTestResult(0) && x != SelfTestResult(1))
      throw RandomErr(Name() + ": Incorrect result with seed " +
                      g.SeedString());
    seed_type s[] = {0x1234U, 0x5678U, 0x9abcU, 0xdef0U};
    //    seed_type s[] = {1, 2, 3, 4};
    g.Reseed(s, s+4);
    g.StepCount(-20000);
    std::string save;
    {
      std::ostringstream stream;
      stream << g << "\n";
      save = stream.str();
    }
    g.Reset();
    {
      std::istringstream stream(save);
      stream >> g;
    }
    g.SetCount(10000);
    {
      std::ostringstream stream;
      g.Save(stream, true);
      save = stream.str();
    }
    {
      std::istringstream stream(save);
      RandomEngine h(std::vector<seed_type>(0));
      h.Load(stream, true);
      h.SetCount(1000000-1);
      x = h();
      if (SelfTestResult(0) && x != SelfTestResult(2))
        throw RandomErr(Name() + ": Incorrect result with seed " +
                        h.SeedString());
      g.SetCount(1000000);
      if (h != g)
        throw RandomErr(Name() + ": Comparison failure");
    }
  }

  template<> Random_u32::type
  RandomEngine<MT19937<Random_u32>, MixerMT0<Random_u32> >::
  SelfTestResult(unsigned i) throw() {
    return i == 0 ? 1 :
      i == 1 ? 4123659995UL : 3016432305UL;
  }

  template<> Random_u64::type
  RandomEngine<MT19937<Random_u64>, MixerMT0<Random_u64> >::
  SelfTestResult(unsigned i) throw() {
    return i == 0 ? 1 :
      i == 1 ? 9981545732273789042ULL : 1384037754719008581ULL;
  }

  template<> Random_u32::type
  RandomEngine<MT19937<Random_u32>, MixerMT1<Random_u32> >::
  SelfTestResult(unsigned i) throw() {
    return i == 0 ? 1 :
      i == 1 ? 4123659995UL : 2924523180UL;
  }

  template<> Random_u64::type
  RandomEngine<MT19937<Random_u64>, MixerMT1<Random_u64> >::
  SelfTestResult(unsigned i) throw() {
    return i == 0 ? 1 :
      i == 1 ? 9981545732273789042ULL : 5481486777409645478ULL;
  }

  template<> Random_u32::type
  RandomEngine<MT19937<Random_u32>, MixerSFMT>::
  SelfTestResult(unsigned i) throw() {
    return i == 0 ? 1 :
      i == 1 ? 666528879UL : 2183745132UL;
  }

  template<> Random_u64::type
  RandomEngine<MT19937<Random_u64>, MixerSFMT>::
  SelfTestResult(unsigned i) throw() {
    return i == 0 ? 1 :
      i == 1 ? 12176471137395770412ULL : 66914054428611861ULL;
  }

  template<> Random_u32::type
  RandomEngine<SFMT19937<Random_u32>, MixerSFMT>::
  SelfTestResult(unsigned i) throw() {
    return i == 0 ? 1 :
      i == 1 ? 2695024307UL : 782200760UL;
  }

  template<> Random_u64::type
  RandomEngine<SFMT19937<Random_u64>, MixerSFMT>::
  SelfTestResult(unsigned i) throw() {
    return i == 0 ? 1 :
      i == 1 ? 1464461649847485149ULL : 5050640804923595109ULL;
  }

  // RandomMixer implementation

  template<class RandomType> void MixerMT0<RandomType>::
  SeedToState(const std::vector<RandomSeed::seed_type>& seed,
              mixer_type state[], unsigned n) throw() {
    // Adapted from
    // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
    // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/VERSIONS/C-LANG/mt19937-64.c
    const unsigned s = unsigned(seed.size());
    const unsigned w = mixer_t::width;

    mixer_type r = s ? a1 + mixer_type(0) : a0 + mixer_type(0);
    state[0] = r;
    for (unsigned k = 1; k < n; ++k) {
      r = b * (r ^ r >> (w - 2)) + k;
      r &= mask;
      state[k] = r;
    }
    if (s > 0) {
      const unsigned m = mixer_t::width / 32,
        s2 = (s + m - 1)/m;
      unsigned i1 = 1;
      r = state[0];
      for (unsigned k = (n > s2 ? n : s2), j = 0;
           k; --k, i1 = i1 == n - 1 ? 1 : i1 + 1, // i1 = i1 + 1 mod n - 1
             j = j == s2 - 1 ? 0 : j + 1 ) {      // j = j+1 mod s2
        r = state[i1] ^ c * (r ^ r >> (w - 2));
        r += j + mixer_type(seed[m * j]) +
          (m == 1 || 2 * j + 1 == s ? mixer_type(0) :
           mixer_type(seed[m * j + 1]) << (w - 32));
        r &= mask;
        state[i1] = r;
      }
      for (unsigned k = n - 1; k; --k,
             i1 = i1 == n - 1 ? 1 : i1 + 1) { // i1 = i1 + 1 mod n - 1
        r = state[i1] ^ d * (r ^ r >> (w - 2));
        r -= i1;
        r &= mask;
        state[i1] = r;
      }
      state[0] = typename mixer_t::type(1) << (w - 1);
    }
  }

  template<class RandomType> void MixerMT1<RandomType>::
  SeedToState(const std::vector<RandomSeed::seed_type>& seed,
              mixer_type state[], unsigned n) throw() {
    // This is the algorithm given in the seed_seq class described in
    // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2079.pdf It is
    // a modification of
    // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
    // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/VERSIONS/C-LANG/mt19937-64.c
    const unsigned s = unsigned(seed.size());
    const unsigned w = mixer_t::width;

    mixer_type r = (a + s) & mask;
    state[0] = r;
    for (unsigned k = 1; k < n; ++k) {
      r = b * (r ^ r >> (w - 2)) + k;
      r &= mask;
      state[k] = r;
    }
    if (s > 0) {
      const unsigned m = mixer_t::width / 32,
        s2 = (s + m - 1)/m;
      unsigned i1 = 0;
      for (unsigned k = (n > s2 ? n : s2), j = 0;
           k; --k, i1 = i1 == n - 1 ? 0 : i1 + 1, // i1 = i1 + 1 mod n
             j = j == s2 - 1 ? 0 : j + 1 ) {      // j = j+1 mod s2
        r = state[i1] ^ c * (r ^ r >> (w - 2));
        r += j + mixer_type(seed[m * j]) +
          (m == 1 || 2 * j + 1 == s ? mixer_type(0) :
           mixer_type(seed[m * j + 1]) << (w - 32));
        r &= mask;
        state[i1] = r;
      }
      for (unsigned k = n; k; --k,
             i1 = i1 == n - 1 ? 0 : i1 + 1) { // i1 = i1 + 1 mod n
        r = state[i1] ^ d * (r ^ r >> (w - 2));
        r -= i1;
        r &= mask;
        state[i1] = r;
      }
    }
  }

  void MixerSFMT::SeedToState(const std::vector<RandomSeed::seed_type>& seed,
                              mixer_type state[], unsigned n) throw() {
    // This is adapted from the routine init_by_array by Mutsuo Saito given in
    // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/SFMT-src-1.2.tar.gz

    if (n == 0)
      return;                   // Nothing to do

    const unsigned s = unsigned(seed.size()),
      // Add treatment of small n with lag = (n - 1)/2 for n <= 7.  In
      // particular, the first operation (xor or plus) in each for loop
      // involves three distinct indices for n > 2.
      lag = n >= 623 ? 11 : (n >= 68 ? 7 : (n >= 39 ? 5 :
                                            (n >= 7 ? 3 : (n - 1)/2))),
      // count = max( s + 1, n )
      count = s + 1 > n ? s + 1 : n;

    std::fill(state, state + n, mixer_type(a));
    const unsigned w = mixer_t::width;

    unsigned i = 0, k = (n - lag) / 2, l = k + lag;
    mixer_type r = state[n - 1];
    for (unsigned j = 0; j < count; ++j,
           i = i == n - 1 ? 0 : i + 1,
           k = k == n - 1 ? 0 : k + 1,
           l = l == n - 1 ? 0 : l + 1) {
      // Here r = state[(j - 1) mod n]
      //      i = j mod n
      //      k = (j + (n - lag)/2) mod n
      //      l = (j + (n - lag)/2 + lag) mod n
      r ^= state[i] ^ state[k];
      r &= mask;
      r = b * (r ^ r >> (w - 5));
      state[k] += r;
      r += i + (j > s ? 0 : (j ? mixer_type(seed[j - 1]) : s));
      state[l] += r;
      state[i] = r;
    }

    for (unsigned j = n; j; --j,
           i = i == n - 1 ? 0 : i + 1,
           k = k == n - 1 ? 0 : k + 1,
           l = l == n - 1 ? 0 : l + 1) {
      // Here r = state[(i - 1) mod n]
      //      k = (i + (n - lag)/2) mod n
      //      l = (i + (n - lag)/2 + lag) mod n
      r += state[i] + state[k];
      r &= mask;
      r = c * (r ^ r >> (w - 5));
      r &= mask;
      state[k] ^= r;
      r -= i;
      r &= mask;
      state[l] ^= r;
      state[i] = r;
    }
  }

  // RandomAlgorithm implementation

  // Here, input is I, J = I + 1, K = I + M; output is I = I + N (mod N)

#define MT19937_STEP(I, J, K) statev[I] = statev[K] ^       \
    (statev[J] & engine_type(1) ? magic : engine_type(0)) ^ \
    ((statev[I] & upper) | (statev[J] & lower)) >> 1

  // The code is cleaned up a little from Hagita's Fortran version by getting
  // rid of the unnecessary masking by YMASK and by using simpler logic to
  // restore the correct value of _state[0].
  //
  // Here input is J = I + N - 1, K = I + M - 1, and p = y[I] (only the high
  // bits are used); output _state[I] and p = y[I - 1].

#define MT19937_REVSTEP(I, J, K) {                                  \
    engine_type q = statev[J] ^ statev[K], s = q >> (width - 1);    \
    q = (q ^ (s ? magic : engine_type(0))) << 1 | s;                \
    statev[I] = (p & upper) | (q & lower);                          \
    p = q;                                                          \
  }

  template<class RandomType>
  void MT19937<RandomType>::Transition(long long count, internal_type statev[])
    throw() {
    if (count > 0)
      for (; count; --count) {
        // This ONLY uses high bit of statev[0]
        unsigned i = 0;
        for (; i < N - M; ++i) MT19937_STEP(i, i + 1, i + M    );
        for (; i < N - 1; ++i) MT19937_STEP(i, i + 1, i + M - N);
        MT19937_STEP(N - 1, 0, M - 1); // i = N - 1
      }
    else if (count < 0)
      for (; count; ++count) {
        // This ONLY uses high bit of statev[0]
        engine_type p = statev[0];
        // Fix low bits of statev[0] and compute y[-1]
        MT19937_REVSTEP(0, N - 1, M - 1); // i = N
        unsigned i = N - 1;
        for (; i > N - M; --i) MT19937_REVSTEP(i, i - 1, i + M - 1 - N);
        for (; i        ; --i) MT19937_REVSTEP(i, i - 1, i + M - 1    );
        MT19937_REVSTEP(0, N - 1, M - 1); // i = 0
      }
  }

#undef MT19937_STEP
#undef MT19937_REVSTEP

  template<class RandomType>
  void MT19937<RandomType>::NormalizeState(engine_type state[]) throw() {

    // Perform the MT-specific sanity check on the resulting state ensuring
    // that the significant 19937 bits are not all zero.
    state[0] &= upper;          // Mask out unused bits
    unsigned i = 0;
    while (i < N && state[i] == 0)
      ++i;
    if (i >= N)
      state[0] = engine_type(1) << (width - 1); // with prob 2^-19937

    // This sets the low R bits of _state[0] consistent with the rest of the
    // state.  Needed to handle SetCount(-N); Ran32(); immediately following
    // reseeding.  This wasn't required in the original code because a
    // Transition was always done first.
    engine_type q = state[N - 1] ^ state[M - 1], s = q >> (width - 1);
    q = (q ^ (s ? magic : engine_type(0))) << 1 | s;
    state[0] = (state[0] & upper) | (q & lower);
  }

  template<class RandomType>
  void MT19937<RandomType>::CheckState(const engine_type state[],
                                       Random_u32::type& check) {
    engine_type x = 0;
    Random_u32::type c = check;
    for (unsigned i = 0; i < N; ++i) {
      engine_t::CheckSum(state[i], c);
      x |= state[i];
    }
    if (x == 0)
      throw RandomErr("MT19937: All-zero state");

    // There are only width*(N-1) + 1 = 19937 independent bits of state.  Thus
    // the low width-1 bits of _state[0] are derivable from the other bits in
    // state.  Verify that the redundant bits bits are consistent.
    engine_type q = state[N - 1] ^ state[M - 1], s = q >> (width - 1);
    q = (q ^ (s ? magic : engine_type(0))) << 1 | s;
    if ((q ^ state[0]) & lower)
      throw RandomErr("MT19937: Invalid state");

    check = c;
  }

#if defined(HAVE_SSE2) && HAVE_SSE2

  // Transition is from Saito's Master's Thesis
  // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/M062821.pdf
  //
  // This implements
  //
  //     w_{i+N} = w_i A xor w_M B xor w_{i+N-2} C xor w_{i+N-1} D
  //
  // where w_i is a 128-bit word and
  //
  //     w A = (w << 8)_128 xor w
  //     w B = (w >> 11)_32 & MSK
  //     w C = (w >> 8)_128
  //     w D = (w << 18)_32
  //
  // Here the _128 means shift is of whole 128-bit word.  _32 means the shifts
  // are independently done on each 32-bit word.
  //
  // In SFMT19937_STEP32 and SFMT19937_STEP64 input is I, J = I + M and output
  // is I = I + N (mod N).  On input, s and r give state for I + N - 2 and I +
  // N - 1; on output s and r give state for I + N - 1 and I + N.  The
  // implementation of 128-bit operations is open-coded in a portable fashion
  // (with LSB ordering).
  //
  // N.B. Here N and M are the lags in units of BitWidth words and so are 4
  // (for u32 implementation) or 2 (for u64 implementation) times bigger than
  // defined in Saito's thesis.

  // This is adapted from SFMT-sse.c in the SFMT 1.2 distribution.
  // The order of instructions has been rearranged to increase the
  // speed slightly

#define SFMT19937_STEP128(I, J) {                   \
    internal_type x = _mm_load_si128(statev + I),   \
      y = _mm_srli_epi32(statev[J], 11),            \
      z = _mm_srli_si128(s, 1);                     \
    s = _mm_slli_epi32(r, 18);                      \
    z = _mm_xor_si128(z, x);                        \
    x = _mm_slli_si128(x, 1);                       \
    z = _mm_xor_si128(z, s);                        \
    y = _mm_and_si128(y, m);                        \
    z = _mm_xor_si128(z, x);                        \
    s = r;                                          \
    r = _mm_xor_si128(z, y);                        \
    _mm_store_si128(statev + I, r);                 \
  }

  // This undoes SFMT19937_STEP.  Trivially, we have
  //
  //     w_i A = w_{i+N} xor w_{i+M} B xor w_{i+N-2} C xor w_{i+N-1} D
  //
  // Given w_i A we can determine w_i from the observation that A^16 =
  // identity, thus
  //
  //     w_i = (w_i A) A^15
  //
  // Because x A^(2^n) = x << (8*2^n) xor x, the operation y = x A^15 can be
  // implemented as
  //
  //     y'   = (x    << 64)_128 xor x    = x    A^8
  //     y''  = (y'   << 32)_128 xor y'   = y'   A^4 = x A^12
  //     y''' = (y''  << 16)_128 xor y''  = y''  A^2 = x A^14
  //     y    = (y''' <<  8)_128 xor y''' = y''' A   = x A^15
  //
  // Here input is I = I + N, J = I + M, K = I + N - 2, L = I + N -1, and
  // output is I = I.
  //
  // This is about 15-35% times slower than SFMT19937_STEPNN, because (1) there
  // doesn't appear to be a straightforward way of saving intermediate results
  // across calls as with SFMT19937_STEPNN and (2) w A^15 is slower to compute
  // than w A.

#define SFMT19937_REVSTEP128(I, J, K, L) {              \
    internal_type x = _mm_load_si128(statev + I),       \
      y = _mm_srli_epi32(statev[J], 11),                \
      z = _mm_slli_epi32(statev[L], 18);                \
    y = _mm_and_si128(y, m);                            \
    x = _mm_xor_si128(x, _mm_srli_si128(statev[K], 1)); \
    x = _mm_xor_si128(x, z);                            \
    x = _mm_xor_si128(x, y);                            \
    x = _mm_xor_si128(_mm_slli_si128(x, 8), x);         \
    x = _mm_xor_si128(_mm_slli_si128(x, 4), x);         \
    x = _mm_xor_si128(_mm_slli_si128(x, 2), x);         \
    x = _mm_xor_si128(_mm_slli_si128(x, 1), x);         \
    _mm_store_si128(statev + I, x);                     \
  }

  template<class RandomType>
  void SFMT19937<RandomType>::Transition(long long count,
                                         internal_type statev[])
    throw() {
    const internal_type m = _mm_set_epi32(magic3, magic2, magic1, magic0);
    if (count > 0) {
      internal_type s = _mm_load_si128(statev + N128 - 2),
        r = _mm_load_si128(statev + N128 - 1);
      for (; count; --count) {
        unsigned i = 0;
        for (; i + M128 < N128; ++i) SFMT19937_STEP128(i, i + M128       );
        for (; i < N128       ; ++i) SFMT19937_STEP128(i, i + M128 - N128);
      }
    } else if (count < 0)
      for (; count; ++count) {
        unsigned i = N128;
        for (; i + M128 > N128;) {
          --i; SFMT19937_REVSTEP128(i, i + M128 - N128, i - 2, i - 1);
        }
        for (; i > 2;) {
          --i; SFMT19937_REVSTEP128(i, i + M128, i - 2, i - 1);
        }
        SFMT19937_REVSTEP128(1, M128 + 1, N128 - 1, 0       ); // i = 1
        SFMT19937_REVSTEP128(0, M128    , N128 - 2, N128 - 1); // i = 0
      }
  }

#undef SFMT19937_STEP128
#undef SFMT19937_REVSTEP128

#elif defined(HAVE_ALTIVEC) && HAVE_ALTIVEC

  // The Altivec versions of SFMT19937_{,REV}STEP128 are simply translated from
  // the SSE2 versions.  The only significant differences arise because of the
  // MSB ordering of the PowerPC.  This means that the 32-bit and 64-bit
  // versions are no different because 32-bit and 64-bit words don't pack
  // together in the same way as on an SSE2 machine (see the two definitions of
  // magic).  This also means that the 128-bit byte shifts on an LSB machine
  // change into more complicated byte permutations.

#define ALTIVEC_PERM(X, P) vec_perm(X, P, P)

#define SFMT19937_STEP128(I, J) {                               \
    internal_type x = statev[I],                                \
      z = vec_xor(vec_xor(ALTIVEC_PERM(s, right1), x),          \
                  vec_sl(r, bitleft));                          \
    s = r;                                                      \
    r = vec_xor(z,                                              \
                vec_xor(ALTIVEC_PERM(x, left1),                 \
                        vec_and(vec_sr(statev[J], bitright),    \
                                magic)));                       \
    statev[I] = r;                                              \
  }

#define SFMT19937_REVSTEP128(I, J, K, L) {              \
    internal_type x = statev[I],                        \
      y = vec_sr(statev[J], bitright),                  \
      z = vec_sl(statev[L], bitleft);                   \
    y = vec_and(y, magic);                              \
    x = vec_xor(x, ALTIVEC_PERM(statev[K], right1));    \
    x = vec_xor(x, z);                                  \
    x = vec_xor(x, y);                                  \
    x = vec_xor(ALTIVEC_PERM(x, left8), x);             \
    x = vec_xor(ALTIVEC_PERM(x, left4), x);             \
    x = vec_xor(ALTIVEC_PERM(x, left2), x);             \
    statev[I] = vec_xor(ALTIVEC_PERM(x, left1), x);     \
  }

  template<class RandomType>
  void SFMT19937<RandomType>::Transition(long long count,
                                         internal_type statev[])
    throw() {
    const internal_type magic = width == 32 ?
      (vector unsigned)(magic0, magic1, magic2, magic3) :
      (vector unsigned)(magic1, magic0, magic3, magic2),
      bitleft = (vector unsigned)(18, 18, 18, 18),
      bitright = (vector unsigned)(11, 11, 11, 11);
    // Shift left and right by 1 byte.  Note that vec_perm(X, Y, P) glues X and
    // Y together into a 32-byte quantity and then the 16-byte permutation
    // vector P specifies which bytes to put into the 16-byte output.  We
    // follow here the convention of using Y = P and using the zero entries in
    // P to allow zero bytes to be introduces into the shifted output.  The
    // following describes how the left1 table (32-bit version) is produced:
    //
    // Byte layout of original with LSB ordering
    // 33 32 31 30  23 22 21 20  13 12 11 10  03 02 01 00
    // shift left by 1 byte (z means zeros enter)
    // 32 31 30 23  22 21 20 13  12 11 10 03  02 01 00 zz
    //
    // Rearrange original to LSB order in 4-byte units
    // 03 02 01 00  13 12 11 10  23 22 21 20  33 32 31 30
    // with sequential MSB byte indices
    // 0  1  2  3   4  5  6  7   8  9 10 11  12 13 14 15
    //
    // Rearrange shift left verion to LSB order in 4-byte units
    // 02 01 00 zz  12 11 10 03  22 21 20 13  32 31 30 23
    // with corresponding MSB byte indices
    // 1  2  3  z   5  6  7  0   9 10 11  4  13 14 15  8
    //
    // Replace byte index at x by 16 + index of 0 = 16 + 7 = 23 to give
    // 1  2  3 23   5  6  7  0   9 10 11  4  13 14 15  8
    const vector unsigned char left1 = width == 32 ?
      (vector unsigned char)(1,2,3,23, 5,6,7,0, 9,10,11,4, 13,14,15,8) :
      (vector unsigned char)(1,2,3,4,5,6,7,31, 9,10,11,12,13,14,15,0),
      right1 = width == 32 ?
      (vector unsigned char)(7,0,1,2, 11,4,5,6, 15,8,9,10, 17,12,13,14) :
      (vector unsigned char)(15,0,1,2,3,4,5,6, 17,8,9,10,11,12,13,14);
    if (count > 0) {
      internal_type s = statev[N128 - 2],
        r = statev[N128 - 1];
      for (; count; --count) {
        unsigned i = 0;
        for (; i + M128 < N128; ++i) SFMT19937_STEP128(i, i + M128       );
        for (; i < N128       ; ++i) SFMT19937_STEP128(i, i + M128 - N128);
      }
    } else if (count < 0) {
      // leftN shifts left by N bytes.
      const vector unsigned char left2 = width == 32 ?
        (vector unsigned char)(2,3,22,22, 6,7,0,1, 10,11,4,5, 14,15,8,9) :
        (vector unsigned char)(2,3,4,5,6,7,30,30, 10,11,12,13,14,15,0,1),
        left4 = width == 32 ?
        (vector unsigned char)(20,20,20,20, 0,1,2,3, 4,5,6,7, 8,9,10,11) :
        (vector unsigned char)(4,5,6,7,28,28,28,28, 12,13,14,15,0,1,2,3),
        left8 = (vector unsigned char)(24,24,24,24,24,24,24,24,0,1,2,3,4,5,6,7);
      for (; count; ++count) {
        unsigned i = N128;
        for (; i + M128 > N128;) {
          --i; SFMT19937_REVSTEP128(i, i + M128 - N128, i - 2, i - 1);
        }
        for (; i > 2;) {
          --i; SFMT19937_REVSTEP128(i, i + M128, i - 2, i - 1);
        }
        SFMT19937_REVSTEP128(1, M128 + 1, N128 - 1, 0       ); // i = 1
        SFMT19937_REVSTEP128(0, M128    , N128 - 2, N128 - 1); // i = 0
      }
    }
  }

#undef SFMT19937_STEP128
#undef SFMT19937_REVSTEP128
#undef ALTIVEC_PERM

#else  // neither HAVE_SSE2 or HAVE_ALTIVEC

#define SFMT19937_STEP32(I, J) {                            \
    internal_type t = statev[I] ^ statev[I] << 8 ^          \
      (statev[J] >> 11 & magic0) ^                          \
      (s0 >> 8 | s1 << 24) ^ r0 << 18;                      \
    s0 = r0; r0 = t & mask;                                 \
    t = statev[I + 1] ^                                     \
      (statev[I + 1] << 8 | statev[I] >> 24) ^              \
      (statev[J + 1] >> 11 & magic1) ^                      \
      (s1 >> 8 | s2 << 24) ^ r1 << 18;                      \
    s1 = r1; r1 = t & mask;                                 \
    t = statev[I + 2] ^                                     \
      (statev[I + 2] << 8 | statev[I + 1] >> 24) ^          \
      (statev[J + 2] >> 11 & magic2) ^                      \
      (s2 >> 8 | s3 << 24) ^ r2 << 18;                      \
    s2 = r2; r2 = t & mask;                                 \
    t = statev[I + 3] ^                                     \
      (statev[I + 3] << 8 | statev[I + 2] >> 24) ^          \
      (statev[J + 3] >> 11 & magic3) ^ s3 >> 8 ^ r3 << 18;  \
    s3 = r3; r3 = t & mask;                                 \
    statev[I    ] = r0; statev[I + 1] = r1;                 \
    statev[I + 2] = r2; statev[I + 3] = r3;                 \
  }

#define SFMT19937_REVSTEP32(I, J, K, L) {                   \
    internal_type                                           \
      t0 = (statev[I] ^ (statev[J] >> 11 & magic0) ^        \
            (statev[K] >> 8 | statev[K + 1] << 24) ^        \
            statev[L] << 18) & mask,                        \
      t1 = (statev[I + 1] ^                                 \
            (statev[J + 1] >> 11 & magic1) ^                \
            (statev[K + 1] >> 8 | statev[K + 2] << 24) ^    \
            statev[L + 1] << 18) & mask,                    \
      t2 = (statev[I + 2] ^                                 \
            (statev[J + 2] >> 11 & magic2) ^                \
            (statev[K + 2] >> 8 | statev[K + 3] << 24) ^    \
            statev[L + 2] << 18) & mask,                    \
      t3 = (statev[I + 3] ^                                 \
            (statev[J + 3] >> 11 & magic3) ^                \
            statev[K + 3] >> 8 ^                            \
            statev[L + 3] << 18) & mask;                    \
    t3 ^= t1; t2 ^= t0; t3 ^= t2; t2 ^= t1; t1 ^= t0;       \
    t3 ^= t2 >> 16 | (t3 << 16 & mask);                     \
    t2 ^= t1 >> 16 | (t2 << 16 & mask);                     \
    t1 ^= t0 >> 16 | (t1 << 16 & mask);                     \
    t0 ^=             t0 << 16 & mask;                      \
    statev[I    ] = t0 ^             (t0 << 8 & mask);      \
    statev[I + 1] = t1 ^ (t0 >> 24 | (t1 << 8 & mask));     \
    statev[I + 2] = t2 ^ (t1 >> 24 | (t2 << 8 & mask));     \
    statev[I + 3] = t3 ^ (t2 >> 24 | (t3 << 8 & mask));     \
  }

  template<>
  void SFMT19937<Random_u32>::Transition(long long count,
                                         internal_type statev[])
    throw() {
    if (count > 0) {
      // x[i+N] = g(x[i], x[i+M], x[i+N-2], x[i,N-1])
      internal_type
        s0 = statev[N - 8], s1 = statev[N - 7],
        s2 = statev[N - 6], s3 = statev[N - 5],
        r0 = statev[N - 4], r1 = statev[N - 3],
        r2 = statev[N - 2], r3 = statev[N - 1];
      for (; count; --count) {
        unsigned i = 0;
        for (; i + M < N; i += R) SFMT19937_STEP32(i, i + M    );
        for (; i < N    ; i += R) SFMT19937_STEP32(i, i + M - N);
      }
    } else if (count < 0)
      for (; count; ++count) {
        unsigned i = N;
        for (; i + M > N;) {
          i -= R; SFMT19937_REVSTEP32(i, i + M - N, i - 2 * R, i - R);
        }
        for (; i > 2 * R;) {
          i -= R; SFMT19937_REVSTEP32(i, i + M    , i - 2 * R, i - R);
        }
        SFMT19937_REVSTEP32(R, M + R, N -     R, 0    ); // i = R
        SFMT19937_REVSTEP32(0, M    , N - 2 * R, N - R); // i = 0
      }
  }

#undef SFMT19937_STEP32
#undef SFMT19937_REVSTEP32

#define SFMT19937_STEP64(I, J) {                    \
    internal_type t = statev[I] ^ statev[I] << 8 ^  \
      (statev[J] >> 11 & magic0) ^                  \
      (s0 >> 8 | s1 << 56) ^ (r0 << 18 & mask18);   \
    s0 = r0; r0 = t & mask;                         \
    t = statev[I + 1] ^                             \
      (statev[I + 1] << 8 | statev[I] >> 56) ^      \
      (statev[J + 1] >> 11 & magic1) ^              \
      s1 >> 8 ^ (r1 << 18 & mask18);                \
    s1 = r1; r1 = t & mask;                         \
    statev[I] = r0; statev[I + 1] = r1;             \
  }

  // In combining the left and right shifts to simulate a 128-bit shift we
  // usually use or.  However we can equivalently use xor (e.g., t1 << 8 ^ t0
  // >> 56 instead of t1 ^ t1 << 8 | t0 >> 56) and this speeds up the code if
  // used in some places.

#define SFMT19937_REVSTEP64(I, J, K, L) {                   \
    internal_type                                           \
      t0 = statev[I] ^ (statev[J] >> 11 & magic0) ^         \
      (statev[K] >> 8 | (statev[K + 1] << 56 & mask)) ^     \
      (statev[L] << 18 & mask18),                           \
      t1 = statev[I + 1] ^ (statev[J + 1] >> 11 & magic1) ^ \
      statev[K + 1] >> 8 ^ (statev[L + 1] << 18 & mask18);  \
    t1 ^= t0;                                               \
    t1 ^= t0 >> 32 ^ (t1 << 32 & mask);                     \
    t0 ^=             t0 << 32 & mask;                      \
    t1 ^= t0 >> 48 ^ (t1 << 16 & mask);                     \
    t0 ^=             t0 << 16 & mask;                      \
    statev[I    ] = t0 ^            (t0 << 8 & mask);       \
    statev[I + 1] = t1 ^ t0 >> 56 ^ (t1 << 8 & mask);       \
  }

  template<>
  void SFMT19937<Random_u64>::Transition(long long count,
                                         internal_type statev[])
    throw() {
    // x[i+N] = g(x[i], x[i+M], x[i+N-2], x[i,N-1])
    if (count > 0) {
      internal_type
        s0 = statev[N - 4], s1 = statev[N - 3],
        r0 = statev[N - 2], r1 = statev[N - 1];
      for (; count; --count) {
        unsigned i = 0;
        for (; i + M < N; i += R) SFMT19937_STEP64(i, i + M    );
        for (; i < N    ; i += R) SFMT19937_STEP64(i, i + M - N);
      }
    } else if (count < 0)
      for (; count; ++count) {
        unsigned i = N;
        for (; i + M > N;) {
          i -= R; SFMT19937_REVSTEP64(i, i + M - N, i - 2 * R, i - R);
        }
        for (; i > 2 * R;) {
          i -= R; SFMT19937_REVSTEP64(i, i + M    , i - 2 * R, i - R);
        }
        SFMT19937_REVSTEP64(R, M + R, N -     R, 0    ); // i = R
        SFMT19937_REVSTEP64(0, M    , N - 2 * R, N - R); // i = 0
      }
  }

#undef SFMT19937_STEP64
#undef SFMT19937_REVSTEP64

#endif  // HAVE_SSE2 and HAVE_ALTIVEC

  template<>
  void SFMT19937<Random_u32>::NormalizeState(engine_type state[]) throw() {
    // Carry out the Period Certification for SFMT19937
    engine_type inner = (state[0] & PARITY0) ^ (state[1] & PARITY1) ^
      (state[2] & PARITY2) ^ (state[3] & PARITY3);
    for (unsigned s = 16; s; s >>= 1)
      inner ^= inner >> s;
    STATIC_ASSERT(PARITY_LSB < 32 && PARITY0 & 1u << PARITY_LSB,
                  "inconsistent PARITY_LSB or PARITY0");
    // Now inner & 1 is the parity of the number of 1 bits in w_0 & p.
    if ((inner & 1u) == 0)
      // Change bit of w_0 corresponding to LSB of PARITY
      state[PARITY_LSB >> 5] ^= engine_type(1u) << (PARITY_LSB & 31u);
  }

  template<>
  void SFMT19937<Random_u64>::NormalizeState(engine_type state[]) throw() {
    // Carry out the Period Certification for SFMT19937
    engine_type inner = (state[0] & PARITY0) ^ (state[1] & PARITY1);
    for (unsigned s = 32; s; s >>= 1)
      inner ^= inner >> s;
    STATIC_ASSERT(PARITY_LSB < 64 && PARITY0 & 1u << PARITY_LSB,
                  "inconsistent PARITY_LSB or PARITY0");
    // Now inner & 1 is the parity of the number of 1 bits in w_0 & p.
    if ((inner & 1u) == 0)
      // Change bit of w_0 corresponding to LSB of PARITY
      state[PARITY_LSB >> 6] ^= engine_type(1u) << (PARITY_LSB & 63u);
  }

  template<class RandomType>
  void SFMT19937<RandomType>::CheckState(const engine_type state[],
                                         Random_u32::type& check) {
    engine_type x = 0;
    Random_u32::type c = check;
    for (unsigned i = 0; i < N; ++i) {
      engine_t::CheckSum(state[i], c);
      x |= state[i];
    }
    if (x == 0)
      throw RandomErr("SFMT19937: All-zero state");
    check = c;
  }

  // RandomPower2 implementation

#if RANDOMLIB_POWERTABLE
  // Powers of two.  Just use floats here.  As long as there's no overflow
  // or underflow these are exact.  In particular they can be cast to
  // doubles or long doubles with no error.
  const float RandomPower2::power2[maxpow - minpow + 1] = {
#if RANDOMLIB_LONGDOUBLEPREC > 64
    // It would be nice to be able to use the C99 notation of 0x1.0p-120
    // for 2^-120 here.
    1/1329227995784915872903807060280344576.f, // 2^-120
    1/664613997892457936451903530140172288.f,  // 2^-119
    1/332306998946228968225951765070086144.f,  // 2^-118
    1/166153499473114484112975882535043072.f,  // 2^-117
    1/83076749736557242056487941267521536.f,   // 2^-116
    1/41538374868278621028243970633760768.f,   // 2^-115
    1/20769187434139310514121985316880384.f,   // 2^-114
    1/10384593717069655257060992658440192.f,   // 2^-113
    1/5192296858534827628530496329220096.f,    // 2^-112
    1/2596148429267413814265248164610048.f,    // 2^-111
    1/1298074214633706907132624082305024.f,    // 2^-110
    1/649037107316853453566312041152512.f,     // 2^-109
    1/324518553658426726783156020576256.f,     // 2^-108
    1/162259276829213363391578010288128.f,     // 2^-107
    1/81129638414606681695789005144064.f,      // 2^-106
    1/40564819207303340847894502572032.f,      // 2^-105
    1/20282409603651670423947251286016.f,      // 2^-104
    1/10141204801825835211973625643008.f,      // 2^-103
    1/5070602400912917605986812821504.f,       // 2^-102
    1/2535301200456458802993406410752.f,       // 2^-101
    1/1267650600228229401496703205376.f,       // 2^-100
    1/633825300114114700748351602688.f,        // 2^-99
    1/316912650057057350374175801344.f,        // 2^-98
    1/158456325028528675187087900672.f,        // 2^-97
    1/79228162514264337593543950336.f,         // 2^-96
    1/39614081257132168796771975168.f,         // 2^-95
    1/19807040628566084398385987584.f,         // 2^-94
    1/9903520314283042199192993792.f,          // 2^-93
    1/4951760157141521099596496896.f,          // 2^-92
    1/2475880078570760549798248448.f,          // 2^-91
    1/1237940039285380274899124224.f,          // 2^-90
    1/618970019642690137449562112.f,           // 2^-89
    1/309485009821345068724781056.f,           // 2^-88
    1/154742504910672534362390528.f,           // 2^-87
    1/77371252455336267181195264.f,            // 2^-86
    1/38685626227668133590597632.f,            // 2^-85
    1/19342813113834066795298816.f,            // 2^-84
    1/9671406556917033397649408.f,             // 2^-83
    1/4835703278458516698824704.f,             // 2^-82
    1/2417851639229258349412352.f,             // 2^-81
    1/1208925819614629174706176.f,             // 2^-80
    1/604462909807314587353088.f,              // 2^-79
    1/302231454903657293676544.f,              // 2^-78
    1/151115727451828646838272.f,              // 2^-77
    1/75557863725914323419136.f,               // 2^-76
    1/37778931862957161709568.f,               // 2^-75
    1/18889465931478580854784.f,               // 2^-74
    1/9444732965739290427392.f,                // 2^-73
    1/4722366482869645213696.f,                // 2^-72
    1/2361183241434822606848.f,                // 2^-71
    1/1180591620717411303424.f,                // 2^-70
    1/590295810358705651712.f,                 // 2^-69
    1/295147905179352825856.f,                 // 2^-68
    1/147573952589676412928.f,                 // 2^-67
    1/73786976294838206464.f,                  // 2^-66
    1/36893488147419103232.f,                  // 2^-65
#endif
    1/18446744073709551616.f,   // 2^-64
    1/9223372036854775808.f,    // 2^-63
    1/4611686018427387904.f,    // 2^-62
    1/2305843009213693952.f,    // 2^-61
    1/1152921504606846976.f,    // 2^-60
    1/576460752303423488.f,     // 2^-59
    1/288230376151711744.f,     // 2^-58
    1/144115188075855872.f,     // 2^-57
    1/72057594037927936.f,      // 2^-56
    1/36028797018963968.f,      // 2^-55
    1/18014398509481984.f,      // 2^-54
    1/9007199254740992.f,       // 2^-53
    1/4503599627370496.f,       // 2^-52
    1/2251799813685248.f,       // 2^-51
    1/1125899906842624.f,       // 2^-50
    1/562949953421312.f,        // 2^-49
    1/281474976710656.f,        // 2^-48
    1/140737488355328.f,        // 2^-47
    1/70368744177664.f,         // 2^-46
    1/35184372088832.f,         // 2^-45
    1/17592186044416.f,         // 2^-44
    1/8796093022208.f,          // 2^-43
    1/4398046511104.f,          // 2^-42
    1/2199023255552.f,          // 2^-41
    1/1099511627776.f,          // 2^-40
    1/549755813888.f,           // 2^-39
    1/274877906944.f,           // 2^-38
    1/137438953472.f,           // 2^-37
    1/68719476736.f,            // 2^-36
    1/34359738368.f,            // 2^-35
    1/17179869184.f,            // 2^-34
    1/8589934592.f,             // 2^-33
    1/4294967296.f,             // 2^-32
    1/2147483648.f,             // 2^-31
    1/1073741824.f,             // 2^-30
    1/536870912.f,              // 2^-29
    1/268435456.f,              // 2^-28
    1/134217728.f,              // 2^-27
    1/67108864.f,               // 2^-26
    1/33554432.f,               // 2^-25
    1/16777216.f,               // 2^-24
    1/8388608.f,                // 2^-23
    1/4194304.f,                // 2^-22
    1/2097152.f,                // 2^-21
    1/1048576.f,                // 2^-20
    1/524288.f,                 // 2^-19
    1/262144.f,                 // 2^-18
    1/131072.f,                 // 2^-17
    1/65536.f,                  // 2^-16
    1/32768.f,                  // 2^-15
    1/16384.f,                  // 2^-14
    1/8192.f,                   // 2^-13
    1/4096.f,                   // 2^-12
    1/2048.f,                   // 2^-11
    1/1024.f,                   // 2^-10
    1/512.f,                    // 2^-9
    1/256.f,                    // 2^-8
    1/128.f,                    // 2^-7
    1/64.f,                     // 2^-6
    1/32.f,                     // 2^-5
    1/16.f,                     // 2^-4
    1/8.f,                      // 2^-3
    1/4.f,                      // 2^-2
    1/2.f,                      // 2^-1
    1.f,                        // 2^0
    2.f,                        // 2^1
    4.f,                        // 2^2
    8.f,                        // 2^3
    16.f,                       // 2^4
    32.f,                       // 2^5
    64.f,                       // 2^6
    128.f,                      // 2^7
    256.f,                      // 2^8
    512.f,                      // 2^9
    1024.f,                     // 2^10
    2048.f,                     // 2^11
    4096.f,                     // 2^12
    8192.f,                     // 2^13
    16384.f,                    // 2^14
    32768.f,                    // 2^15
    65536.f,                    // 2^16
    131072.f,                   // 2^17
    262144.f,                   // 2^18
    524288.f,                   // 2^19
    1048576.f,                  // 2^20
    2097152.f,                  // 2^21
    4194304.f,                  // 2^22
    8388608.f,                  // 2^23
    16777216.f,                 // 2^24
    33554432.f,                 // 2^25
    67108864.f,                 // 2^26
    134217728.f,                // 2^27
    268435456.f,                // 2^28
    536870912.f,                // 2^29
    1073741824.f,               // 2^30
    2147483648.f,               // 2^31
    4294967296.f,               // 2^32
    8589934592.f,               // 2^33
    17179869184.f,              // 2^34
    34359738368.f,              // 2^35
    68719476736.f,              // 2^36
    137438953472.f,             // 2^37
    274877906944.f,             // 2^38
    549755813888.f,             // 2^39
    1099511627776.f,            // 2^40
    2199023255552.f,            // 2^41
    4398046511104.f,            // 2^42
    8796093022208.f,            // 2^43
    17592186044416.f,           // 2^44
    35184372088832.f,           // 2^45
    70368744177664.f,           // 2^46
    140737488355328.f,          // 2^47
    281474976710656.f,          // 2^48
    562949953421312.f,          // 2^49
    1125899906842624.f,         // 2^50
    2251799813685248.f,         // 2^51
    4503599627370496.f,         // 2^52
    9007199254740992.f,         // 2^53
    18014398509481984.f,        // 2^54
    36028797018963968.f,        // 2^55
    72057594037927936.f,        // 2^56
    144115188075855872.f,       // 2^57
    288230376151711744.f,       // 2^58
    576460752303423488.f,       // 2^59
    1152921504606846976.f,      // 2^60
    2305843009213693952.f,      // 2^61
    4611686018427387904.f,      // 2^62
    9223372036854775808.f,      // 2^63
    18446744073709551616.f,     // 2^64
  };
#endif

  // RandomEngine (and implicitly RandomAlgorithm and RandomMixer)
  // instantiations.  The first 4 (using MixerMT[01]) are not recommended.
  template class RandomEngine<  MT19937<Random_u32>, MixerMT0<Random_u32> >;
  template class RandomEngine<  MT19937<Random_u64>, MixerMT0<Random_u64> >;
  template class RandomEngine<  MT19937<Random_u32>, MixerMT1<Random_u32> >;
  template class RandomEngine<  MT19937<Random_u64>, MixerMT1<Random_u64> >;

  template class RandomEngine<  MT19937<Random_u32>, MixerSFMT>;
  template class RandomEngine<  MT19937<Random_u64>, MixerSFMT>;
  template class RandomEngine<SFMT19937<Random_u32>, MixerSFMT>;
  template class RandomEngine<SFMT19937<Random_u64>, MixerSFMT>;

  // RandomCanonial instantiations

  template<> RandomCanonical<MRandomGenerator32>
  RandomCanonical<MRandomGenerator32>::Global = RandomCanonical();
  template<> RandomCanonical<MRandomGenerator64>
  RandomCanonical<MRandomGenerator64>::Global = RandomCanonical();
  template<> RandomCanonical<SRandomGenerator32>
  RandomCanonical<SRandomGenerator32>::Global = RandomCanonical();
  template<> RandomCanonical<SRandomGenerator64>
  RandomCanonical<SRandomGenerator64>::Global = RandomCanonical();

} // namespace RandomLib
