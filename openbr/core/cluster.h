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
#include <openbr/plugins/openbr_internal.h>

namespace br
{
    typedef QList<int> Cluster; // List of indices into galleries
    typedef QVector<Cluster> Clusters;

    // generate k-NN graph from pre-computed similarity matrices 
    Neighborhood knnFromSimmat(const QStringList &simmats, int k = 20);
    Neighborhood knnFromSimmat(const QList<cv::Mat> &simmats, int k = 20);
 
    // Generate k-NN graph from a gallery, using the current algorithm for comparison.
    // direct serialization to file system.
    void  knnFromGallery(const QString &galleryName, const QString & outFile, int k = 20);
    // in memory graph computation
    Neighborhood knnFromGallery(const QString &gallery, int k = 20);

    // Load k-NN graph from a file with the following ascii format:
    // One line per sample, each line lists the top k neighbors for the sample as follows:
    // index1:score1,index2:score2,...,indexk:scorek
    Neighborhood loadkNN(const QString &fname);

    // Save k-NN graph to file
    bool savekNN(const Neighborhood &neighborhood, const QString &outfile);

    // Rank-order clustering on a pre-computed k-NN graph
    Clusters ClusterGraph(Neighborhood neighbors, float aggresssiveness, const QString &csv = "");
    Clusters ClusterGraph(const QString & knnName, float aggressiveness, const QString &csv = "");

    // Given a similarity matrix, compute the k-NN graph, then perform rank-order clustering.
    Clusters ClusterSimmat(const QList<cv::Mat> &simmats, float aggressiveness, const QString &csv = "");
    Clusters ClusterSimmat(const QStringList &simmats, float aggressiveness, const QString &csv = "");

    // evaluate clustering results in csv, reading ground truth data from gallery input, using truth_property
    // as the key for ground truth labels.
    void EvalClustering(const QString &clusters, const QString &truth_gallery, QString truth_property, bool cluster_csv, QString cluster_property);

    // Read/write clusters from a text format, 1 line = 1 cluster, each line contains comma separated list
    // of assigned indices.
    Clusters ReadClusters(const QString &csv);
    void WriteClusters(const Clusters &clusters, const QString &csv);
}

#endif // BR_CLUSTER_H
