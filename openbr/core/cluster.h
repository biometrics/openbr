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

#ifndef BR_CLUSTER_H
#define BR_CLUSTER_H

#include <QList>
#include <QString>
#include <QStringList>
#include <QVector>
#include <openbr/openbr_plugin.h>

namespace br
{
    typedef QList<int> Cluster; // List of indices into galleries
    typedef QVector<Cluster> Clusters;

    Clusters ClusterGallery(const QList<cv::Mat> &simmats, float aggressiveness);
    Clusters ClusterGallery(const QStringList &simmats, float aggressiveness, const QString &csv);
    void EvalClustering(const QString &csv, const QString &input, QString truth_property);

    Clusters ReadClusters(const QString &csv);
    void WriteClusters(const Clusters &clusters, const QString &csv);
}

#endif // BR_CLUSTER_H
