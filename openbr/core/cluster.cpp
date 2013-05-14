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

#include <QDebug>
#include <QFile>
#include <QHash>
#include <QPair>
#include <QSet>
#include <limits>
#include <openbr/openbr_plugin.h>

#include "openbr/core/bee.h"
#include "openbr/core/cluster.h"

typedef QPair<int,float> Neighbor; // QPair<id,similarity>
typedef QList<Neighbor> Neighbors;
typedef QVector<Neighbors> Neighborhood;

// Compare function used to order neighbors from highest to lowest similarity
static bool compareNeighbors(const Neighbor &a, const Neighbor &b)
{
    if (a.second == b.second)
        return a.first < b.first;
    return a.second > b.second;
}

// Zhu et al. "A Rank-Order Distance based Clustering Algorithm for Face Tagging", CVPR 2011
// Ob(x) in eq. 1, modified to consider 0/1 as ground truth imposter/genuine.
static int indexOf(const Neighbors &neighbors, int i)
{
    for (int j=0; j<neighbors.size(); j++) {
        const Neighbor &neighbor = neighbors[j];
        if (neighbor.first == i) {
            if      (neighbor.second == 0) return neighbors.size()-1;
            else if (neighbor.second == 1) return 0;
            else                           return j;
        }
    }
    return -1;
}

// Zhu et al. "A Rank-Order Distance based Clustering Algorithm for Face Tagging", CVPR 2011
// Corresponds to eq. 1, or D(a,b)
static int asymmetricalROD(const Neighborhood &neighborhood, int a, int b)
{
    int distance = 0;
    foreach (const Neighbor &neighbor, neighborhood[a]) {
        if (neighbor.first == b) break;
        int index = indexOf(neighborhood[b], neighbor.first);
        distance += (index == -1) ? neighborhood[b].size() : index;
    }
    return distance;
}

// Zhu et al. "A Rank-Order Distance based Clustering Algorithm for Face Tagging", CVPR 2011
// Corresponds to eq. 2/4, or D-R(a,b)
float normalizedROD(const Neighborhood &neighborhood, int a, int b)
{
    int indexA = indexOf(neighborhood[b], a);
    int indexB = indexOf(neighborhood[a], b);

    // Default behaviors
    if ((indexA == -1) || (indexB == -1)) return std::numeric_limits<float>::max();
    if ((neighborhood[b][indexA].second == 1) || (neighborhood[a][indexB].second == 1)) return 0;
    if ((neighborhood[b][indexA].second == 0) || (neighborhood[a][indexB].second == 0)) return std::numeric_limits<float>::max();

    int distanceA = asymmetricalROD(neighborhood, a, b);
    int distanceB = asymmetricalROD(neighborhood, b, a);
    return 1.f * (distanceA + distanceB) / std::min(indexA+1, indexB+1);
}

Neighborhood getNeighborhood(const QStringList &simmats)
{
    Neighborhood neighborhood;

    float globalMax = -std::numeric_limits<float>::max();
    float globalMin = std::numeric_limits<float>::max();
    int numGalleries = (int)sqrt((float)simmats.size());
    if (numGalleries*numGalleries != simmats.size())
        qFatal("Incorrect number of similarity matrices.");

    // Process each simmat
    for (int i=0; i<numGalleries; i++) {
        QVector<Neighbors> allNeighbors;

        int currentRows = -1;
        int columnOffset = 0;
        for (int j=0; j<numGalleries; j++) {
            cv::Mat m = BEE::readSimmat(simmats[i*numGalleries+j]);
            if (j==0) {
                currentRows = m.rows;
                allNeighbors.resize(currentRows);
            }
            if (currentRows != m.rows) qFatal("Row count mismatch.");

            // Get data row by row
            for (int k=0; k<m.rows; k++) {
                Neighbors &neighbors = allNeighbors[k];
                neighbors.reserve(neighbors.size() + m.cols);
                for (int l=0; l<m.cols; l++) {
                    float val = m.at<float>(k,l);
                    if ((i==j) && (k==l)) continue; // Skips self-similarity scores

                    if ((val != -std::numeric_limits<float>::infinity()) &&
                        (val != std::numeric_limits<float>::infinity())) {
                        globalMax = std::max(globalMax, val);
                        globalMin = std::min(globalMin, val);
                    }
                    neighbors.append(Neighbor(l+columnOffset, val));
                }
            }

            columnOffset += m.cols;
        }

        // Keep the top matches
        for (int j=0; j<allNeighbors.size(); j++) {
            Neighbors &val = allNeighbors[j];
            const int cutoff = 20; // Somewhat arbitrary number of neighbors to keep
            int keep = std::min(cutoff, val.size());
            std::partial_sort(val.begin(), val.begin()+keep, val.end(), compareNeighbors);
            neighborhood.append((Neighbors)val.mid(0, keep));
        }
    }

    // Normalize scores
    for (int i=0; i<neighborhood.size(); i++) {
        Neighbors &neighbors = neighborhood[i];
        for (int j=0; j<neighbors.size(); j++) {
            Neighbor &neighbor = neighbors[j];
            if (neighbor.second == -std::numeric_limits<float>::infinity())
                neighbor.second = 0;
            else if (neighbor.second == std::numeric_limits<float>::infinity())
                neighbor.second = 1;
            else
                neighbor.second = (neighbor.second - globalMin) / (globalMax - globalMin);
        }
    }

    return neighborhood;
}

// Zhu et al. "A Rank-Order Distance based Clustering Algorithm for Face Tagging", CVPR 2011
br::Clusters br::ClusterGallery(const QStringList &simmats, float aggressiveness, const QString &csv)
{
    qDebug("Clustering %d simmat(s)", simmats.size());

    // Read in gallery parts, keeping top neighbors of each template
    Neighborhood neighborhood = getNeighborhood(simmats);
    const int cutoff = neighborhood.first().size();
    const float threshold = 3*cutoff/4 * aggressiveness/5;

    // Initialize clusters
    Clusters clusters(neighborhood.size());
    for (int i=0; i<neighborhood.size(); i++)
        clusters[i].append(i);

    bool done = false;
    while (!done) {
        // nextClusterIds[i] = j means that cluster i is set to merge into cluster j
        QVector<int> nextClusterIDs(neighborhood.size());
        for (int i=0; i<neighborhood.size(); i++) nextClusterIDs[i] = i;

        // For each cluster
        for (int clusterID=0; clusterID<neighborhood.size(); clusterID++) {
            const Neighbors &neighbors = neighborhood[clusterID];
            int nextClusterID = nextClusterIDs[clusterID];

            // Check its neighbors
            foreach (const Neighbor &neighbor, neighbors) {
                int neighborID = neighbor.first;
                int nextNeighborID = nextClusterIDs[neighborID];

                // Don't bother if they have already merged
                if (nextNeighborID == nextClusterID) continue;

                // Flag for merge if similar enough
                if (normalizedROD(neighborhood, clusterID, neighborID) < threshold) {
                    if (nextClusterID < nextNeighborID) nextClusterIDs[neighborID] = nextClusterID;
                    else                                nextClusterIDs[clusterID] = nextNeighborID;
                }
            }
        }

        // Transitive merge
        for (int i=0; i<neighborhood.size(); i++) {
            int nextClusterID = i;
            while (nextClusterID != nextClusterIDs[nextClusterID]) {
                assert(nextClusterIDs[nextClusterID] < nextClusterID);
                nextClusterID = nextClusterIDs[nextClusterID];
            }
            nextClusterIDs[i] = nextClusterID;
        }

        // Construct new clusters
        QHash<int, int> clusterIDLUT;
        QList<int> allClusterIDs = QSet<int>::fromList(nextClusterIDs.toList()).values();
        for (int i=0; i<neighborhood.size(); i++)
            clusterIDLUT[i] = allClusterIDs.indexOf(nextClusterIDs[i]);

        Clusters newClusters(allClusterIDs.size());
        Neighborhood newNeighborhood(allClusterIDs.size());

        for (int i=0; i<neighborhood.size(); i++) {
            int newID = clusterIDLUT[i];
            newClusters[newID].append(clusters[i]);
            newNeighborhood[newID].append(neighborhood[i]);
        }

        // Update indices and trim
        for (int i=0; i<newNeighborhood.size(); i++) {
            Neighbors &neighbors = newNeighborhood[i];
            int size = qMin(neighbors.size(),cutoff);
            std::partial_sort(neighbors.begin(), neighbors.begin()+size, neighbors.end(), compareNeighbors);
            for (int j=0; j<size; j++)
                neighbors[j].first = clusterIDLUT[j];
            neighbors = neighbors.mid(0, cutoff);
        }

        // Update results
        done = true; //(newClusters.size() >= clusters.size());
        clusters = newClusters;
        neighborhood = newNeighborhood;
    }

    // Save clusters
    if (!csv.isEmpty())
        WriteClusters(clusters, csv);
    return clusters;
}

// Santo Fortunato "Community detection in graphs", Physics Reports 486 (2010)
// wI or wII metric (page 148)
float wallaceMetric(const br::Clusters &clusters, const QVector<int> &indices)
{
    int matches = 0;
    int total = 0;
    foreach (const QList<int> &cluster, clusters) {
        for (int i=0; i<cluster.size(); i++) {
            for (int j=i+1; j<cluster.size(); j++) {
                total++;
                if (indices[cluster[i]] == indices[cluster[j]])
                    matches++;
            }
        }
    }
    return (float)matches/(float)total;
}

// Santo Fortunato "Community detection in graphs", Physics Reports 486 (2010)
// Jaccard index (page 149)
float jaccardIndex(const QVector<int> &indicesA, const QVector<int> &indicesB)
{
    int a[2][2] = {{0,0},{0,0}};
    for (int i=0; i<indicesA.size()-1; i++)
        for (int j=i+1; j<indicesA.size(); j++)
            a[indicesA[i] == indicesA[j] ? 1 : 0][indicesB[i] == indicesB[j] ? 1 : 0]++;

    return float(a[1][1]) / (a[0][1] + a[1][0] + a[1][1]);
}

// Evaluates clustering algorithms based on metrics described in
// Santo Fortunato "Community detection in graphs", Physics Reports 486 (2010)
void br::EvalClustering(const QString &csv, const QString &input)
{
    qDebug("Evaluating %s against %s", qPrintable(csv), qPrintable(input));

    QList<float> labels = TemplateList::fromGallery(input).files().collectValues<float>("Label");

    QHash<int, int> labelToIndex;
    int nClusters = 0;
    for (int i=0; i<labels.size(); i++) {
        const float &label = labels[i];
        if (!labelToIndex.contains(label))
            labelToIndex[label] = nClusters++;
    }

    Clusters truthClusters; truthClusters.reserve(nClusters);
    for (int i=0; i<nClusters; i++)
        truthClusters.append(QList<int>());

    QVector<int> truthIndices(labels.size());
    for (int i=0; i<labels.size(); i++) {
        truthIndices[i] = labelToIndex[labels[i]];
        truthClusters[labelToIndex[labels[i]]].append(i);
    }

    Clusters testClusters = ReadClusters(csv);

    QVector<int> testIndices(labels.size());
    for (int i=0; i<testClusters.size(); i++)
        for (int j=0; j<testClusters[i].size(); j++)
            testIndices[testClusters[i][j]] = i;

    // At this point the following 4 things are defined:
    // truthClusters - list of clusters of template_ids based on subject_ids
    // truthIndices - template_id to cluster_id based on sigset subject_ids
    // testClusters - list of clusters of template_ids based on csv input
    // testIndices - template_id to cluster_id based on testClusters

    float wI = wallaceMetric(truthClusters, testIndices);
    float wII = wallaceMetric(testClusters, truthIndices);
    float jaccard = jaccardIndex(testIndices, truthIndices);
    qDebug("Recall: %f  Precision: %f  F-score: %f  Jaccard index: %f", wI, wII, sqrt(wI*wII), jaccard);
}

br::Clusters br::ReadClusters(const QString &csv)
{
    Clusters clusters;
    QFile file(csv);
    bool success = file.open(QFile::ReadOnly);
    if (!success) qFatal("Failed to open %s for reading.", qPrintable(csv));
    QStringList lines = QString(file.readAll()).split("\n");
    file.close();

    foreach (const QString &line, lines) {
        Cluster cluster;
        QStringList ids = line.trimmed().split(",", QString::SkipEmptyParts);
        foreach (const QString &id, ids) {
            bool ok;
            cluster.append(id.toInt(&ok));
            if (!ok) qFatal("Non-interger id.");
        }
        clusters.append(cluster);
    }
    return clusters;
}

void br::WriteClusters(const Clusters &clusters, const QString &csv)
{
    QFile file(csv);
    bool success = file.open(QFile::WriteOnly);
    if (!success) qFatal("Failed to open %s for writing.", qPrintable(csv));

    foreach (Cluster cluster, clusters) {
        if (cluster.empty()) continue;

        qSort(cluster);
        QStringList ids;
        foreach (int id, cluster)
            ids.append(QString::number(id));
        file.write(qPrintable(ids.join(",")+"\n"));
    }
    file.close();
}
