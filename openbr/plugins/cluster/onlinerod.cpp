#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/cluster.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \brief Constructors clusters based on the Rank-Order distance in an online, incremental manner
 * \author Charles Otto \cite caotto
 * \author Jordan Cheney \cite JordanCheney
 * \br_property br::Distance* distance Distance to compute the similarity score between templates. Default is L2.
 * \br_property int kNN Maximum number of nearest neighbors to keep for each template. Default is 20.
 * \br_property float aggression Clustering aggresiveness. A higher value will result in larger clusters. Default is 10.
 * \br_property bool incremental If true, compute the clusters as each template is processed otherwise compute the templates at the end. Default is false.
 * \br_property QString evalOutput Path to store cluster informtation. Optional. Default is an empty string.
 * \br_paper Zhu et al.
 *           "A Rank-Order Distance based Clustering Algorithm for Face Tagging"
 *           CVPR 2011
 */
class OnlineRODTransform : public TimeVaryingTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Distance *distance READ get_distance WRITE set_distance RESET reset_distance STORED true)
    Q_PROPERTY(int kNN READ get_kNN WRITE set_kNN RESET reset_kNN STORED false)
    Q_PROPERTY(float aggression READ get_aggression WRITE set_aggression RESET reset_aggression STORED false)
    Q_PROPERTY(bool incremental READ get_incremental WRITE set_incremental RESET reset_incremental STORED false)
    Q_PROPERTY(QString evalOutput READ get_evalOutput WRITE set_evalOutput RESET reset_evalOutput STORED false)

    BR_PROPERTY(br::Distance*, distance, Distance::make(".Dist(L2)",this))
    BR_PROPERTY(int, kNN, 20)
    BR_PROPERTY(float, aggression, 10)
    BR_PROPERTY(bool, incremental, false)
    BR_PROPERTY(QString, evalOutput, "")

    TemplateList templates;
    Neighborhood neighborhood;

public:
    OnlineRODTransform() : TimeVaryingTransform(false, false) {}

private:
    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        // update current graph
        foreach(const Template &t, src) {
            QList<float> scores = distance->compare(templates, t);

            // attempt to udpate each existing point's (sorted) k-NN list with these results.
            Neighbors currentN;
            for (int i=0; i < scores.size(); i++) {
                currentN.append(Neighbor(i, scores[i]));
                Neighbors target = neighborhood[i];

                // should we insert the new neighbor into the current target's list?
                if (target.size() < kNN || scores[i] > target.last().second) {
                    // insert into the sorted nearest neighbor list
                    Neighbor temp(scores.size(), scores[i]);
                    br::Neighbors::iterator res = qLowerBound(target.begin(), target.end(), temp, compareNeighbors);
                    target.insert(res, temp);

                    if (target.size() > kNN)
                        target.removeLast();

                    neighborhood[i] = target;
                }
            }

            // add a new row, consisting of the top neighbors of the newest point
            int actuallyKeep = std::min(kNN, currentN.size());
            std::partial_sort(currentN.begin(), currentN.begin()+actuallyKeep, currentN.end(), compareNeighbors);

            Neighbors selected = currentN.mid(0, actuallyKeep);
            neighborhood.append(selected);
            templates.append(t);
        }

        if (incremental)
            identifyClusters(dst);
    }

    void finalize(TemplateList &output)
    {
        if (!templates.empty()) {
            identifyClusters(output);
            templates.clear();
            neighborhood.clear();
        }
    }

    void identifyClusters(TemplateList &dst)
    {
        Clusters clusters = ClusterGraph(neighborhood, aggression, evalOutput);
        if (Globals->verbose)
            qDebug("Built %d clusters from %d templates", clusters.size(), templates.size());

        for (int i = 0; i < clusters.size(); i++) {
            // Calculate the centroid of each cluster
            Mat center = Mat::zeros(templates[0].m().rows, templates[0].m().cols, CV_32F);
            foreach (int t, clusters[i]) {
                Mat converted; templates[t].m().convertTo(converted, CV_32F);
                center += converted;
            }
            center /= clusters[i].size();

            // Calculate the Euclidean distance from the center to use as the cluster confidence
            foreach (int t, clusters[i]) {
                templates[t].file.set("Cluster", i);
                Mat c; templates[t].m().convertTo(c, CV_32F);
                Mat p; pow(c - center, 2, p);
                templates[t].file.set("ClusterConfidence", sqrt(sum(p)[0]));
            }
        }
        dst.append(templates);
    }
};

BR_REGISTER(Transform, OnlineRODTransform)

} // namespace br

#include "onlinerod.moc"

