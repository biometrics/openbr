/*
  tanh_sse.h

  example for approximation of hyperbolic tangent from Lambert's (Gauss)
  continued fraction.
  http://en.wikipedia.org/wiki/Gauss's_continued_fraction
  http://maths.ashwyninnovations.com/lambert.pdf

  plain c and sse intrinsic versions for msvc and gcc

  compile with:
    gcc -msse -mfpmath=387 -W -Wall
    gcc -msse -mfpmath=387,sse -W -Wall
    cl /arch:SSE /W4

  notes:
    - the plain c version also ends up quite fast with gcc's fpu optimizations
    - _TANH_RANGE could be brought down, so that the "clamp" engages for lower
      values. error starts to propagate near 5,6 or -5,-6 in the 0.00001 range
    - if _TANH_CLAMP_INDIVIDUAL is not set the entire vector will be clamped
    - speed comparison against libm on amd althon xp with gcc 4.x for one and
      the same value in the vector:
        flags:          -O3 -msse -mfpmath=387
        iterations:     1E+6
        fast_tanh_sse:  37 ms
        libm:           172 ms
    - orders of 5/6 are used:
      tanh(x) = (21*x^5 + 1260*x^3 + 10395*x) /
                (x^6 + 210*x^4 + 4725*x^2 + 10395)
    - define _TANH_FAST_DIV for less accurate (~14% faster) division with
      ~(21 - 22) bits of mantissa.

  contact:
    lubomir i. ivanov, neolit123 [at] gmail
*/

#ifndef _INCLUDE_TANH_SSE_
#define _INCLUDE_TANH_SSE_

/* inline */
#ifdef _TANH_USE_INLINE
  #define _TANH_INLINE    inline
#else
  #define _TANH_INLINE
#endif

/* constants */
#define _TANH_RANGE   5.f
#define _TANH_CLAMP   1.f
#define _TANH_K0      21.f
#define _TANH_K1      210.f
#define _TANH_K2      1260.f
#define _TANH_K3      4725.f
#define _TANH_K4      10395.f

/* types */
#define v4sf          __m128

/* portable c version */
_TANH_INLINE
float
fast_tanh(const float x)
{
  const     float s = x*x;
  register  float d;

  if      (x < -_TANH_RANGE)
    return -_TANH_CLAMP;
  else if (x > _TANH_RANGE)
    return _TANH_CLAMP;

  d =     (s*(s*(s + _TANH_K1) + _TANH_K3) + _TANH_K4);
  return  (x*(s*(_TANH_K0*s + _TANH_K2) + _TANH_K4)) / d;
}

/* sse intrinsic version */
#ifdef __SSE__  // jklontz 1/17/2012
#include "xmmintrin.h"

_TANH_INLINE
v4sf
fast_tanh_sse(const v4sf x)
{
  v4sf    y, s, d;
  #ifdef _TANH_FAST_DIV
    v4sf  i_d;
  #endif

  /* check each value in the vector */
  #ifdef _TANH_CLAMP_INDIVIDUAL
    register short i;
    union
    {
      v4sf    v;
      float   f[4];
    } u;

    i = 0;
    u.v = x;
    while (i < 4)
    {
      if (u.f[i] < -_TANH_RANGE)
        u.f[i] = -_TANH_CLAMP;
      if (u.f[i] > _TANH_RANGE)
        u.f[i] = _TANH_CLAMP;
      i++;
    }
  /* clamp entire vector */
  #else
    if      (_mm_movemask_ps(_mm_cmplt_ps(x, _mm_set1_ps(-_TANH_RANGE))))
      return _mm_set1_ps(-_TANH_CLAMP);
    else if (_mm_movemask_ps(_mm_cmpgt_ps(x, _mm_set1_ps( _TANH_RANGE))))
      return _mm_set1_ps( _TANH_CLAMP);
  #endif

  s = _mm_mul_ps(x, x);

  /* denominator */
  d = _mm_add_ps(s, _mm_set1_ps(_TANH_K1));
  d = _mm_mul_ps(d, s);
  d = _mm_add_ps(d, _mm_set1_ps(_TANH_K3));
  d = _mm_mul_ps(d, s);
  d = _mm_add_ps(d, _mm_set1_ps(_TANH_K4));

  /* numerator */
  y = _mm_mul_ps(s, _mm_set1_ps(_TANH_K0));
  y = _mm_add_ps(y, _mm_set1_ps(_TANH_K2));
  y = _mm_mul_ps(y, s);
  y = _mm_add_ps(y, _mm_set1_ps(_TANH_K4));
  y = _mm_mul_ps(y, x);

  #ifdef _TANH_FAST_DIV
    /* reciprocal of the denominator with one newton iteration */
    i_d = _mm_rcp_ps(d);
    i_d = _mm_sub_ps( _mm_add_ps(i_d, i_d),
                      _mm_mul_ps(d, _mm_mul_ps(i_d, i_d)));
    return _mm_mul_ps(y, i_d);
  #else
    return _mm_div_ps(y, d);
  #endif
}
#endif

#endif /* _INCLUDE_TANH_SSE_ */
