/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef DISTANCE_SSE_H
#define DISTANCE_SSE_H

#include <QDebug>

#ifdef __SSE__

#include <xmmintrin.h>

inline QDebug operator<<(QDebug dbg, const __m128i &p)
{
    uchar *up = (uchar*)&p;
    dbg.nospace() << up[0] << "," << up[1] << "," << up[2] << "," << up[3] << "," << up[4] << "," << up[5] << "," << up[6] << "," << up[7] << ","
                  << up[8] << "," << up[9] << "," << up[10] << "," << up[11] << "," << up[12] << "," << up[13] << "," << up[14] << "," << up[15];
    return dbg.space();
}

inline float l1(const uchar *a, const uchar *b, int size)
{
    size = size / sizeof(__m128i);
    __m128i accumulate = _mm_setzero_si128();

    for (int i=0; i<size; i++) {
        __m128i A = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a)+i);
        __m128i B = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b)+i);
        __m128i sad = _mm_sad_epu8(A, B);
        accumulate = _mm_add_epi64(sad, accumulate);
    }

    int64_t buff[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&buff), accumulate);
    return buff[0] + buff[1];
}

#else

inline float l1(const uchar *a, const uchar *b, int size)
{
    int distance = 0;
    for (int i=0; i<size; i++)
        distance += abs(a[i]-b[i]);
    return distance;
}

#endif

inline float packed_l1(const uchar *a, const uchar *b, int size)
{
    static const uchar low_mask = 0x0F;
    static const uchar hi_mask = 0xF0;

    int distance = 0;
    for (int i=0; i<size; i++)
        distance += (abs((a[i] & low_mask) - (b[i] & low_mask)) >> 0) +
                    (abs((a[i] & hi_mask)  - (b[i] & hi_mask))  >> 4);
    return distance;
}

#endif // DISTANCE_SSE_H
