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

#ifndef BR_PLOT_H
#define BR_PLOT_H

#include <QPair>
#include <QString>
#include <QVector>
#include <openbr/openbr_plugin.h>

namespace br
{
    bool Plot(const QStringList &files, const File &destination, bool show = false);
    bool PlotDetection(const QStringList &files, const File &destination, bool show = false);
    bool PlotLandmarking(const QStringList &files, const File &destination, bool show = false);
    bool PlotMetadata(const QStringList &files, const QString &destination, bool show = false);
    bool PlotKNN(const QStringList &files, const File &destination, bool show = false);
    bool PlotEER(const QStringList &files, const File &destination, bool show = false);
}

#endif // BR_PLOT_H
