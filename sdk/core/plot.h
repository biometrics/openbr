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

#ifndef __PLOT_H
#define __PLOT_H

#include <QPair>
#include <QString>
#include <QVector>

namespace br
{

void Confusion(const QString &file, float score, int &true_positives, int &false_positives, int &true_negatives, int &false_negatives);
float Evaluate(const QString &simmat, const QString &mask, const QString &csv = ""); // Returns TAR @ FAR = 0.01
bool Plot(const QStringList &files, const QString &destination, bool show = false);
bool PlotMetadata(const QStringList &files, const QString &destination, bool show = false);

}

#endif // __PLOT_H
