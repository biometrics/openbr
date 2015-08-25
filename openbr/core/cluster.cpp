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
#include <assert.h>

#include "openbr/core/bee.h"
#include "openbr/core/cluster.h"
#include "openbr/plugins/openbr_internal.h"

using namespace br;

// Compare function used to order neighbors from highest to lowest similarity
bool br::compareNeighbors(const Neighbor &a, const Neighbor &b)
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

Neighborhood br::knnFromSimmat(const QList<cv::Mat> &simmats, int k)
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
            cv::Mat m = simmats[i * numGalleries + j];
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

                    if (val != -std::numeric_limits<float>::max()
                        && val != -std::numeric_limits<float>::infinity()
                        && val != std::numeric_limits<float>::infinity()) {
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
            const int cutoff = k; // Number of neighbors to keep
            int keep = std::min(cutoff, val.size());
            std::partial_sort(val.begin(), val.begin()+keep, val.end(), compareNeighbors);
            neighborhood.append((Neighbors)val.mid(0, keep));
        }
    }

    return neighborhood;
}

// generate k-NN graph from pre-computed similarity matrices 
Neighborhood br::knnFromSimmat(const QStringList &simmats, int k)
{
    QList<cv::Mat> mats;
    foreach (const QString &simmat, simmats) {
        QScopedPointer<br::Format> format(br::Factory<br::Format>::make(simmat));
        br::Template t = format->read();
        mats.append(t);
    }
    return knnFromSimmat(mats, k);
}

TemplateList knnFromGallery(const QString & galleryName, bool inMemory, const QString & outFile, int k)
{
    QSharedPointer<Transform> comparison = Transform::fromComparison(Globals->algorithm);

    Gallery *tempG = Gallery::make(galleryName);
    qint64 total = tempG->totalSize();
    delete tempG;
    comparison->setPropertyRecursive("galleryName", galleryName+"[dropMetadata=true]");

    bool multiProcess = Globals->file.getBool("multiProcess", false);
    if (multiProcess)
        comparison = QSharedPointer<Transform> (br::wrapTransform(comparison.data(), "ProcessWrapper"));

    QScopedPointer<Transform> collect(Transform::make("CollectNN+ProgressCounter+Discard", NULL));
    collect->setPropertyRecursive("totalProgress", total);
    collect->setPropertyRecursive("keep", k);

    QList<Transform *> tforms;
    tforms.append(comparison.data());
    tforms.append(collect.data());

    QScopedPointer<Transform> compareCollect(br::pipeTransforms(tforms));

    QSharedPointer <Transform> projector;
    if (inMemory)
        projector = QSharedPointer<Transform> (br::wrapTransform(compareCollect.data(), "Stream(readMode=StreamGallery, endPoint=Discard"));
    else
        projector = QSharedPointer<Transform> (br::wrapTransform(compareCollect.data(), "Stream(readMode=StreamGallery, endPoint=LogNN("+outFile+")+DiscardTemplates)"));

    TemplateList input;
    input.append(Template(galleryName));
    TemplateList output;

    projector->init();
    projector->projectUpdate(input, output);

    return output;
}
 
// Generate k-NN graph from a gallery, using the current algorithm for comparison.
// Direct serialization to file system, k-NN graph is not retained in memory
void br::knnFromGallery(const QString &galleryName, const QString &outFile, int k)
{
    knnFromGallery(galleryName, false, outFile, k);
}

// In-memory graph construction
Neighborhood br::knnFromGallery(const QString &gallery, int k)
{
    // Nearest neighbor data current stored as template metadata, so retrieve it
    TemplateList res = knnFromGallery(gallery, true, "", k);

    Neighborhood neighborhood;
    foreach (const Template &t, res) {
        Neighbors neighbors = t.file.get<Neighbors>("neighbors");
        neighbors.append(neighbors);
    }

    return neighborhood;
}

Neighborhood br::loadkNN(const QString &infile)
{
    Neighborhood neighborhood;
    QFile file(infile);
    bool success = file.open(QFile::ReadOnly);
    if (!success) qFatal("Failed to open %s for reading.", qPrintable(infile));
    QStringList lines = QString(file.readAll()).split("\n");
    file.close();
    int min_idx = INT_MAX;
    int max_idx = -1;
    int count = 0;

    foreach (const QString &line, lines) {
        Neighbors neighbors;
        count++;
        if (line.trimmed().isEmpty()) {
            neighborhood.append(neighbors);
            continue;
        }

        QStringList list = line.trimmed().split(",", QString::SkipEmptyParts);
        foreach (const QString &item, list) {
            QStringList parts = item.trimmed().split(":", QString::SkipEmptyParts);
            bool intOK = true;
            bool floatOK = true;
            int idx = parts[0].toInt(&intOK);
            float score = parts[1].toFloat(&floatOK);

            if (idx > max_idx)
                max_idx = idx;
            if (idx  <min_idx)
                min_idx = idx;

            if (idx >= lines.size())
                continue;
            neighbors.append(qMakePair(idx, score));

            if (!intOK && floatOK)
                qFatal("Failed to parse word: %s", qPrintable(item));
        }
        neighborhood.append(neighbors);
    }
    return neighborhood;
}

bool br::savekNN(const Neighborhood &neighborhood, const QString &outfile)
{
    QFile file(outfile);
    bool success = file.open(QFile::WriteOnly);
    if (!success) qFatal("Failed to open %s for writing.", qPrintable(outfile));

    foreach (Neighbors neighbors, neighborhood) {
        QString aLine;
        if (!neighbors.empty())
        {
            aLine.append(QString::number(neighbors[0].first)+":"+QString::number(neighbors[0].second));
            for (int i=1; i < neighbors.size();i++) {
                aLine.append(","+QString::number(neighbors[i].first)+":"+QString::number(neighbors[i].second));
            }
        }
        aLine += "\n";
        file.write(qPrintable(aLine));
    }
    file.close();
    return true;
}


// Rank-order clustering on a pre-computed k-NN graph
Clusters br::ClusterGraph(Neighborhood neighborhood, float aggressiveness, const QString &csv)
{

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

    if (!csv.isEmpty())
        WriteClusters(clusters, csv);

    return clusters;
}

Clusters br::ClusterGraph(const QString & knnName, float aggressiveness, const QString &csv)
{
    Neighborhood neighbors = loadkNN(knnName);
    return ClusterGraph(neighbors, aggressiveness, csv);
}

// Zhu et al. "A Rank-Order Distance based Clustering Algorithm for Face Tagging", CVPR 2011
br::Clusters br::ClusterSimmat(const QList<cv::Mat> &simmats, float aggressiveness, const QString &csv)
{
    qDebug("Clustering %d simmat(s), aggressiveness %f", simmats.size(), aggressiveness);

    // Read in gallery parts, keeping top neighbors of each template
    Neighborhood neighborhood = knnFromSimmat(simmats);

    return ClusterGraph(neighborhood, aggressiveness, csv);
}

br::Clusters br::ClusterSimmat(const QStringList &simmats, float aggressiveness, const QString &csv)
{
    QList<cv::Mat> mats;
    foreach (const QString &simmat, simmats) {
        QScopedPointer<br::Format> format(br::Factory<br::Format>::make(simmat));
        br::Template t = format->read();
        mats.append(t);
    }

    Clusters clusters = ClusterSimmat(mats, aggressiveness, csv);
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

// Cluster purity => Assign each cluster a class based on the majority ground truth value,
//                   calculate percentage of correct assignments
// http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html#fig:clustfg3
float purityMetric(const br::Clusters &clusters, const QVector<int> &truthIdx)
{
    float correct = 0.0;
    int total = 0;
    foreach(const Cluster &c, clusters) {
        total += c.size();
        QList<int> truthVals;
        foreach(int templateID, c) {
            truthVals.append(truthIdx[templateID]);
        }
        int max = 0;
        foreach(int clustID, truthVals.toSet()) {
            int cnt = truthVals.count(clustID);
            if (cnt > max) {
                max = cnt;
            }
        }
        correct += max;
    }
    return correct/total;
}

// Evaluates clustering algorithms based on metrics described in
// Santo Fortunato "Community detection in graphs", Physics Reports 486 (2010)
void br::EvalClustering(const QString &clusters, const QString &truth_gallery, QString truth_property, bool cluster_csv, QString cluster_property)
{
    if (truth_property.isEmpty())
        truth_property = "Label";
    if (!cluster_csv && cluster_property.isEmpty()) {
        cluster_property = "ClusterID";
    }
    qDebug("Evaluating %s against %s", qPrintable(clusters), qPrintable(truth_gallery));

    TemplateList tList = TemplateList::fromGallery(truth_gallery);
    QList<int> labels = tList.indexProperty(truth_property);

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

    Clusters testClusters;
    if (cluster_csv) {
        testClusters = ReadClusters(clusters);
    } else {
        // get Clusters from gallery
        const TemplateList tl(TemplateList::fromGallery(clusters));
        QHash<int,Cluster > clust_id_map;
        for (int i=0; i<tl.size(); i++) {
            const Template &t = tl.at(i);
            int c = t.file.get<int>(cluster_property);
            if (!clust_id_map.contains(c)) {
                clust_id_map.insert(c, Cluster());
            }
            clust_id_map[c].append(i);
        }
        testClusters = Clusters::fromList(clust_id_map.values());
    }

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
    float purity = purityMetric(testClusters, truthIndices);
    qDebug("Purity: %f  Recall: %f  Precision: %f  F-score: %f  Jaccard index: %f", purity, wI, wII, sqrt(wI*wII), jaccard);
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
