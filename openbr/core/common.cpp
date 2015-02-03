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

#include "common.h"
#include <QMutex>
#include <RandomLib/Random.hpp>

using namespace std;

static RandomLib::Random g_rand;
static QMutex rngLock;

/**** GLOBAL ****/
void Common::seedRNG() {
    QMutexLocker lock(&rngLock);

    static bool seeded = false;
    if (!seeded) {
        srand(0); // We seed with 0 instead of time(NULL) to have reproducible randomness
        seeded = true;
        g_rand.Reseed(0);
    }
}

double Common::randN()
{
    QMutexLocker lock(&rngLock);

    return g_rand.FloatN();
}

QList<int> Common::RandSample(int n, int max, int min, bool unique)
{
    QList<int> samples; samples.reserve(n);
    int range = max-min;
    if (range <= 0) qFatal("Non-positive range.");
    if (unique && (n >= range)) {
        for (int i=min; i<max; i++)
            samples.append(i);
        return samples;
    }

    while (samples.size() < n) {
        const int sample = (rand() % range) + min;
        if (unique && samples.contains(sample)) continue;
        samples.append(sample);
    }
    return samples;
}

QList<int> Common::RandSample(int n, const QSet<int> &values, bool unique)
{
    QList<int> valueList = values.toList();
    if (unique && (values.size() <= n)) return valueList;

    QList<int> samples; samples.reserve(n);
    while (samples.size() < n) {
        const int randIndex = rand() % valueList.size();
        samples.append(valueList[randIndex]);
        if (unique) valueList.removeAt(randIndex);
    }
    return samples;
}

QList<float> Common::linspace(float start, float stop, int n) {
    float delta = (stop - start) / (n - 1);
    float curValue = start;
    QList<float> spaced;
    spaced.reserve(n);
    spaced.append(start);
    for (int i = 1; i < (n - 1); i++) {
        spaced.append(curValue += delta);
    }
    spaced.append(stop);
    return spaced;
}

QList<int> Common::ind2sub(int dims, int nPerDim, int idx) {
    QList<int> subIndices;
    for (int j = 0; j < dims; j++) {
        subIndices.append(((int)floor( idx / pow((float)nPerDim, j))) % nPerDim);
    }
    return subIndices;
}
