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

#ifndef COMMON_COMMON_H
#define COMMON_COMMON_H

#include <QDebug>
#include <QList>
#include <QMap>
#include <QPair>
#include <QSet>
#include <QtAlgorithms>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <assert.h>
#include <math.h>
#include <time.h>

namespace Common
{

/*!
 * \brief Round floating point to nearest integer.
 */
template <typename T>
int round(T r)
{
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}

/*!
 * \brief Returns a list of pairs sorted by value where:
 *        pair.first = original value
 *        pair.second = original index
 */
template <typename T>
QList< QPair<T,int> > Sort(const QList<T> &vals, bool decending = false, int n = std::numeric_limits<int>::max())
{
    const int size = vals.size();
    QList< QPair<T,int> > pairs; pairs.reserve(size);
    for (int i=0; i<size; i++) pairs.append(QPair<T,int>(vals[i], i));

    if (n >= pairs.size()) {
        if (decending) std::sort(pairs.begin(), pairs.end(), std::greater< QPair<T,int> >());
        else           std::sort(pairs.begin(), pairs.end(), std::less< QPair<T,int> >());
    } else {
        if (decending) std::partial_sort(pairs.begin(), pairs.begin()+n, pairs.end(), std::greater< QPair<T,int> >());
        else           std::partial_sort(pairs.begin(), pairs.begin()+n, pairs.end(), std::less< QPair<T,int> >());
        pairs = pairs.mid(0, n);
    }

    return pairs;
}

/*!
 * \brief Returns the minimum, maximum, minimum index, and maximum index of a vector of values.
 */
template <template<class> class V, typename T>
void MinMax(const V<T> &vals, T *min, T *max, int *min_index, int *max_index)
{
    const int size = vals.size();
    assert(size > 0);

    *min = *max = vals[0];
    *min_index = *max_index = 0;
    for (int i=1; i<size; i++) {
        const T val = vals[i];
        if (val < *min) {
            *min = val;
            *min_index = i;
        } else if (val > *max) {
            *max = val;
            *max_index = i;
        }
    }
}

template <template<class> class V, typename T>
void MinMax(const V<T> &vals, T *min, T *max)
{
    int min_index, max_index;
    MinMax(vals, min, max, &min_index, &max_index);
}

template <template<class> class V, typename T>
T Min(const V<T> &vals)
{
    T min, max;
    MinMax(vals, &min, &max);
    return min;
}

template <template<class> class V, typename T>
T Max(const V<T> &vals)
{
    T min, max;
    MinMax(vals, &min, &max);
    return max;
}

/*!
 * \brief Returns the sum of a vector of values.
 */
template <template<class> class V, typename T>
double Sum(const V<T> &vals)
{
    double sum = 0;
    foreach (T val, vals) sum += val;
    return sum;
}

/*!
 * \brief Returns the mean and standard deviation of a vector of values.
 */
template <template<class> class V, typename T>
double Mean(const V<T> &vals)
{
    if (vals.isEmpty()) return 0;
    return Sum(vals) / vals.size();
}

/*!
 * \brief Returns the mean and standard deviation of a vector of values.
 */
template <template<class> class V, typename T>
void MeanStdDev(const V<T> &vals, double *mean, double *stddev)
{
    *mean = Mean(vals);
    if (vals.isEmpty()) {
        *stddev = 0;
        return;
    }

    double variance = 0;
    foreach (T val, vals) {
        const double delta = val - *mean;
        variance += delta * delta;
    }
    *stddev = sqrt(variance/vals.size());
}

/*!
 * \brief Computes the median of a list.
 */
template<template<typename> class C, typename T>
T Median(C<T> vals, T *q1 = 0, T *q3 = 0)
{
    qSort(vals);
    if (q1 != 0) *q1 = vals.isEmpty() ? 0 : vals[1*vals.size()/4];
    if (q3 != 0) *q3 = vals.isEmpty() ? 0 : vals[3*vals.size()/4];
    return vals.isEmpty() ? 0 : vals[vals.size()/2];
}

/*!
 * \brief Computes the mode of a list.
 */
template <typename T>
T Mode(const QList<T> &vals)
{
    QMap<T,int> counts;
    foreach (const T &val, vals) {
        if (!counts.contains(val))
            counts[val] = 0;
        counts[val]++;
    }
    return counts.key(Max(counts.values()));
}

/*!
 * \brief Returns the cumulative sum of a vector of values.
 */
template <typename T>
QList<T> CumSum(const QList<T> &vals)
{
    QList<T> cumsum;
    cumsum.reserve(vals.size()+1);
    cumsum.append(0);
    foreach (const T &val, vals)
        cumsum.append(cumsum.last()+val);
    return cumsum;
}

/*!
 * \brief Calculate DKE bandwidth parameter 'h'
 */
template <template<class> class V, typename T>
double KernelDensityBandwidth(const V<T> &vals)
{
    double mean, stddev;
    MeanStdDev(vals, &mean, &stddev);
    return pow(4 * pow(stddev, 5.0) / (3 * vals.size()), 0.2);
}

/*!
 * \brief Compute kernel density at value x with bandwidth h.
 */
template <template<class> class V, typename T>
double KernelDensityEstimation(const V<T> &vals, double x, double h)
{
    double y = 0;
    foreach (T val, vals)
        y += exp(-pow((val-x)/h,2)/2)/sqrt(2*3.1415926353898);
    return y / (vals.size() * h);
}

// Return a random number, uniformly distributed over 0,1
double randN();

/*!
 * \brief Returns a vector of n integers sampled in the range <min, max].
 *
 * If unique then there will be no repeated integers.
 * \note Algorithm is inefficient for unique vectors where n ~= max-min.
 */
void seedRNG();
QList<int> RandSample(int n, int max, int min = 0, bool unique = false);
QList<int> RandSample(int n, const QSet<int> &values, bool unique = false);

/*!
 * \brief Weighted random sample, each entry in weights should be >= 0.
 */
template <typename T>
QList<int> RandSample(int n, const QList<T> &weights, bool unique = false)
{
    QList<T> cdf = CumSum(weights);
    for (int i=0; i<cdf.size(); i++) // Normalize cdf
        cdf[i] = cdf[i] / cdf.last();

    QList<int> samples; samples.reserve(n);
    while (samples.size() < n) {
        T r = randN();

        for (int j=0; j<weights.size(); j++) {
            if ((r >= cdf[j]) && (r <= cdf[j+1])) {
                if (!unique || !samples.contains(j))
                    samples.append(j);
                break;
            }
        }
    }

    return samples;
}

/*!
 * \brief See Matlab function linspace() for documentation.
 */
QList<float> linspace(float start, float stop, int n);

/*!
 * \brief See Matlab function unique() for documentation.
 */
template <typename T>
void Unique(const QList<T> &vals, QList<T> &b, QList<int> &m, QList<int> &n)
{
    const int size = vals.size();
    assert(size > 0);

    b.reserve(size);
    m.reserve(size);
    n.reserve(size);

    // Compute b and m
    QList< QPair<T, int> > sortedPairs = Sort(vals);
    b.append(sortedPairs[0].first);
    m.append(sortedPairs[0].second);
    for (size_t i=1; i<size; i++) {
        if (sortedPairs[i].first == b.back()) {
            m.back() = qMax(m.back(), sortedPairs[i].second);
        } else {
            b.append(sortedPairs[i].first);
            m.append(sortedPairs[i].second);
        }
    }

    // Compute n
    for (int i=0; i<size; i++) n.append(b.indexOf(vals[i]));
}

/*!
 * \brief Given a vector of pairs, constructs two new vectors from pair.first and pair.second.
 */
template <typename T, typename U>
void SplitPairs(const QList< QPair<T,U> > &pairs, QList<T> &first, QList<U> &second)
{
    first.reserve(pairs.size());
    second.reserve(pairs.size());
    typedef QPair<T,U> pair_t;
    foreach (const pair_t &pair, pairs) {
        first.append(pair.first);
        second.append(pair.second);
    }
}

/*!
 * \brief Removes values outside of 1.5 * Inner Quartile Range.
 */
template <typename T>
QList<T> RemoveOutliers(QList<T> vals)
{
    T q1, q3;
    Median(vals, &q1, &q3);
    T iqr = q3-q1;
    T min = q1 - 1.5*iqr;
    T max = q3 + 1.5*iqr;
    QList<T> newVals;
    for (int i=0; i<vals.size(); i++)
        if ((vals[i] >= min) && (vals[i] <= max))
            newVals.append(vals[i]);
    return newVals;
}

/*!
 * \brief Sorts and evenly downsamples a vector to size k.
 */
template <template<class> class V, typename T>
V<T> Downsample(V<T> vals, int k)
{
    std::sort(vals.begin(), vals.end());
    int size = vals.size();
    if (size <= k) return vals;

    V<T> newVals; newVals.reserve(k);
    for (int i=0; i<k; i++) newVals.push_back(vals[long(i) * long(size-1) / long(k-1)]);
    return newVals;
}

/*! \brief Converts index into subdimensions.
*/
QList<int> ind2sub(int dims, int nPerDim, int idx);

}

#endif // COMMON_COMMON_H
