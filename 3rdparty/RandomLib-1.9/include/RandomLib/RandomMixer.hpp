/**
 * \file RandomMixer.hpp
 * \brief Header for Mixer classes.
 *
 * Mixer classes convert a seed vector into a random generator state.  An
 * important property of this method is that "close" seeds should produce
 * "widely separated" states.  This allows the seeds to be set is some
 * systematic fashion to produce a set of uncorrelated random number
 * sequences.
 *
 * Copyright (c) Charles Karney (2006-2011) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * http://randomlib.sourceforge.net/
 **********************************************************************/

#if !defined(RANDOMLIB_RANDOMMIXER_HPP)
#define RANDOMLIB_RANDOMMIXER_HPP 1

#include <vector>
#include <string>
#include <RandomLib/RandomSeed.hpp>

namespace RandomLib {

  /**
   * \brief The original %MT19937 mixing functionality
   *
   * This implements the functionality of init_by_array in MT19937
   * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
   * and init_by_array64 in MT19937_64
   * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/VERSIONS/C-LANG/mt19937-64.c
   * with the following changes:
   * - in the case of an zero-length seed array, behave in the same way if
   *   MT19937 and MT19937_64 are called without initialization in which case,
   *   e.g., init_genrand(5489UL) is called.  (init_by_array does not allow
   *   calling with a zero-length seed.)
   * - init_by_array64 accepts a seed array of 64-bit unsigned ints.  Here with
   *   seed is an array of 32-bit unsigned ints and these are repacked into
   *   64-bit quantities internally using a LSB convention.  Thus, to mimic the
   *   MT19937_64 sample invocation with a seed array {0x12345ULL, 0x23456ULL,
   *   0x34567ULL, 0x45678ULL}, MixerMT0<Random_u64>::SeedToState needs to
   *   be invoked with a seed vector [0x12345UL, 0, 0x23456UL, 0, 0x34567UL, 0,
   *   0x45678UL, 0].  (Actually the last 0 is unnecessary.)
   *
   * The template parameter \e RandomType switches between the 32-bit and
   * 64-bit versions.
   *
   * MixerMT0 is specific to the MT19937 generators and should not be used
   * for other generators (e.g., SFMT19937).  In addition, MixerMT0 has
   * known defects and should only be used to check the operation of the
   * MT19937 engines against the original implementation.  These defects are
   * described in the MixerMT1 which is a modification of MixerMT0
   * which corrects these defects.  For production use MixerMT1 or,
   * preferably, MixerSFMT should be used.
   *
   * @tparam RandomType the type of the results, either Random_u32 or
   *   Random_u64.
   **********************************************************************/
  template<class RandomType> class RANDOMLIB_EXPORT MixerMT0 {
  public:
    /**
     * The RandomType controlling the output of MixerMT0::SeedToState
     **********************************************************************/
    typedef RandomType mixer_t;
    /**
     * A version number which should be unique to this RandomMixer.  This
     * prevents RandomEngine::Load from loading a saved generator with a
     * different RandomMixer.  Here the version is "MxMT" or "MxMU".
     **********************************************************************/
    static const unsigned version = 0x4d784d54UL + (mixer_t::width == 64);
  private:
    /**
     * The unsigned type corresponding to mixer_t.
     **********************************************************************/
    typedef typename mixer_t::type mixer_type;
    /**
     * The mask for mixer_t.
     **********************************************************************/
    static const mixer_type mask = mixer_t::mask;
  public:
    /**
     * Mix the seed vector, \e seed, into the state array, \e state, of size \e
     * n.
     *
     * @param[in] seed the input seed vector.
     * @param[out] state the generator state.
     * @param[in] n the size of the state.
     **********************************************************************/
    static void SeedToState(const std::vector<RandomSeed::seed_type>& seed,
                            mixer_type state[], unsigned n) throw();
    /**
     * Return the name of this class.
     *
     * @return the name.
     **********************************************************************/
    static std::string Name() {
      return "MixerMT0<Random_u" +
        std::string(mixer_t::width == 32 ? "32" : "64") + ">";
    }
  private:
    static const mixer_type a0 = 5489ULL;
    static const mixer_type a1 = 19650218ULL;
    static const mixer_type
      b = mixer_t::width == 32 ? 1812433253ULL : 6364136223846793005ULL;
    static const mixer_type
      c = mixer_t::width == 32 ?    1664525ULL : 3935559000370003845ULL;
    static const mixer_type
      d = mixer_t::width == 32 ? 1566083941ULL : 2862933555777941757ULL;
  };

  /**
   * \brief The modified %MT19937 mixing functionality
   *
   * MixerMT0 has two defects
   * - The zeroth word of the state is set to a constant (independent of the
   *   seed).  This is a relatively minor defect which halves the accessible
   *   state space for MT19937 (but the resulting state space is still huge).
   *   (Actually, for the 64-bit version, it reduces the accessible states by
   *   2<sup>33</sup>.  On the other hand the 64-bit has better mixing
   *   properties.)
   * - Close seeds, for example, [1] and [1,0], result in the same state.  This
   *   is a potentially serious flaw which might result is identical random
   *   number sequences being generated instead of independent sequences.
   *
   * MixerMT1 fixes these defects in a straightforward manner.  The
   * resulting algorithm was included in one of the proposals for Random Number
   * Generation for C++0X, see Brown, et al.,
   * http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2079.pdf
   *
   * The template parameter \e RandomType switches between the 32-bit and
   * 64-bit versions.
   *
   * MixerMT1 still has a weakness in that it doesn't thoroughly mix the
   * state.  This is illustrated by an example given to me by Makoto Matsumoto:
   * Consider a seed of length \e N and suppose we consider all \e
   * W<sup><i>N</i>/2</sup> values for the first half of the seed (here \e W =
   * 2<sup><i>width</i></sup>).  MixerMT1 has a bottleneck in the way that
   * the state is initialized which results in the second half of the state
   * only taking on \e W<sup>2</sup> possible values.  MixerSFMT mixes the
   * seed into the state much more thoroughly.
   *
   * @tparam RandomType the type of the results, either Random_u32 or
   *   Random_u64.
   **********************************************************************/
  template<class RandomType> class RANDOMLIB_EXPORT MixerMT1 {
  public:
    /**
     * The RandomType controlling the output of MixerMT1::SeedToState
     **********************************************************************/
    typedef RandomType mixer_t;
    /**
     * A version number which should be unique to this RandomMixer.  This
     * prevents RandomEngine::Load from loading a saved generator with a
     * different RandomMixer.  Here the version is "MxMV" or "MxMW".
     **********************************************************************/
    static const unsigned version = 0x4d784d56UL + (mixer_t::width == 64);
  private:
    /**
     * The unsigned type corresponding to mixer_t.
     **********************************************************************/
    typedef typename mixer_t::type mixer_type;
    /**
     * The mask for mixer_t.
     **********************************************************************/
    static const mixer_type mask = mixer_t::mask;
  public:
    /**
     * Mix the seed vector, \e seed, into the state array, \e state, of size \e
     * n.
     *
     * @param[in] seed the input seed vector.
     * @param[out] state the generator state.
     * @param[in] n the size of the state.
     **********************************************************************/
    static void SeedToState(const std::vector<RandomSeed::seed_type>& seed,
                            mixer_type state[], unsigned n) throw();
    /**
     * Return the name of this class.
     *
     * @return the name.
     **********************************************************************/
    static std::string Name() {
      return "MixerMT1<Random_u" +
        std::string(mixer_t::width == 32 ? "32" : "64") + ">";
    }
  private:
    static const mixer_type a = 5489ULL;
    static const mixer_type
      b = mixer_t::width == 32 ? 1812433253ULL : 6364136223846793005ULL;
    static const mixer_type
      c = mixer_t::width == 32 ?    1664525ULL : 3935559000370003845ULL;
    static const mixer_type
      d = mixer_t::width == 32 ? 1566083941ULL : 2862933555777941757ULL;
  };

  /**
   * \brief The SFMT mixing functionality
   *
   * MixerSFMT is adapted from SFMT's init_by_array Mutsuo Saito given in
   * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/SFMT-src-1.2.tar.gz
   * and is part of the C++11 standard; see P. Becker, Working Draft, Standard
   * for Programming Language C++, Oct. 2007, Sec. 26.4.7.1,
   * http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2461.pdf
   *
   * MixerSFMT contains a single change is to allow it to function properly
   * when the size of the state is small.
   *
   * MixerSFMT mixes the seed much more thoroughly than MixerMT1 and, in
   * particular, it removes the mixing bottleneck present in MixerMT1.
   * Thus it is the recommended mixing scheme for all production work.
   **********************************************************************/
  class RANDOMLIB_EXPORT MixerSFMT {
  public:
    /**
     * The RandomType controlling the output of MixerSFMT::SeedToState
     **********************************************************************/
    typedef Random_u32 mixer_t;
    /**
     * A version number which should be unique to this RandomMixer.  This
     * prevents RandomEngine::Load from loading a saved generator with a
     * different RandomMixer.  Here the version is "MxSM".
     **********************************************************************/
    static const unsigned version = 0x4d78534dUL;
  private:
    /**
     * The unsigned type corresponding to mixer_t.
     **********************************************************************/
    typedef mixer_t::type mixer_type;
    /**
     * The mask for mixer_t.
     **********************************************************************/
    static const mixer_type mask = mixer_t::mask;
  public:
    /**
     * Mix the seed vector, \e seed, into the state array, \e state, of size \e
     * n.
     *
     * @param[in] seed the input seed vector.
     * @param[out] state the generator state.
     * @param[in] n the size of the state.
     **********************************************************************/
    static void SeedToState(const std::vector<RandomSeed::seed_type>& seed,
                            mixer_type state[], unsigned n) throw();
    /**
     * Return the name of this class.
     *
     * @return the name.
     **********************************************************************/
    static std::string Name() { return "MixerSFMT"; }
  private:
    static const mixer_type a = 0x8b8b8b8bUL;
    static const mixer_type b = 1664525UL;
    static const mixer_type c = 1566083941UL;
  };

} // namespace RandomLib

#endif  // RANDOMLIB_RANDOMMIXER_HPP
