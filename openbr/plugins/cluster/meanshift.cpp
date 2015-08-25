#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \brief A transform implementing the mean shift clustering algorithm.
 * \author Jordan Cheney \cite JordanCheney
 * \br_property br::Distance* distance The distance used to compute the distance between templates
 * \br_property int kernelBandwidth The size of the kernel used to converge points to a cluster center
 * \br_property float shiftThreshold The cutoff threshold distance for a shifted point. A value lower then this threshold indicates a point has finished shifting to a cluster center.
 * \br_property float distanceThreshold The distance threshold for a point to join a cluster. A point must be at least this close to another point to be included in the same cluster as that point.
 * \br_link http://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
 */
class MeanShiftClusteringTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(int kernelBandwidth READ get_kernelBandwidth WRITE set_kernelBandwidth RESET reset_kernelBandwidth STORED false)
    Q_PROPERTY(float shiftThreshold READ get_shiftThreshold WRITE set_shiftThreshold RESET reset_shiftThreshold STORED false)
    Q_PROPERTY(float distanceThreshold READ get_distanceThreshold WRITE set_distanceThreshold RESET reset_distanceThreshold STORED false)
    BR_PROPERTY(br::Distance*, distance, Distance::make(".Dist(L2, false)", NULL))
    BR_PROPERTY(int, kernelBandwidth, 3)
    BR_PROPERTY(float, shiftThreshold, 1e-3)
    BR_PROPERTY(float, distanceThreshold, 1e-1)

public:
    MeanShiftClusteringTransform() : TimeVaryingTransform(false, false) {}

private:
    void projectUpdate(const TemplateList &src, TemplateList &)
    {
        templates.append(src);
    }

    void finalize(TemplateList &output)
    {
        output.clear();

        QList<Mat> original_points, shifted_points;
        original_points = shifted_points = templates.data();

        Mat shift_mask = Mat::zeros(1, shifted_points.size(), CV_32S);
        while (countNonZero(shift_mask) != shifted_points.size()) {
            for (int i = 0; i < shifted_points.size(); i++) {
                if (shift_mask.at<int>(0, i))
                    continue;

                Mat point = shifted_points[i];
                Mat shifted_point = point.clone();
                meanshift(shifted_point, original_points);

                float dist = distance->compare(point, shifted_point);
                if (dist < shiftThreshold)
                    shift_mask.at<int>(0, i) = 1;

                shifted_points[i] = shifted_point;
            }
        }

        QList<int> clusters = assignClusterID(shifted_points);
        for (int i = 0; i < templates.size(); i++)
            templates[i].file.set("Cluster", clusters[i]);
        output.append(templates);
    }

    void meanshift(Mat &point, const QList<Mat> &original_points)
    {
        Mat distances(1, original_points.size(), CV_32FC1);
        for (int i = 0; i < original_points.size(); i++)
            distances.at<float>(0, i) = distance->compare(point, original_points[i]);

        Mat weights = gaussianKernel(distances, kernelBandwidth);
        point = (weights * OpenCVUtils::toMat(original_points)) /  sum(weights)[0];
    }

    inline Mat gaussianKernel(const Mat &distance, const float bandwidth)
    {
        Mat p, e;
        pow(distance / bandwidth, 2, p);
        exp(-0.5 * p, e);

        return (1.0 / (bandwidth * sqrt(2 * CV_PI))) * e;
    }

    QList<int> assignClusterID(const QList<Mat> &points)
    {
        QList<int> groups;
        int newGroupIdx = 0;
        foreach (const Mat &point, points) {
            int group = nearestGroup(point, points, groups);
            if (group < 0)
                group = newGroupIdx++;
            groups.append(group);
        }

        if (Globals->verbose)
            qDebug("created %d clusters from %d templates", newGroupIdx, points.size());

        return groups;
    }

    int nearestGroup(const Mat &point, const QList<Mat> &points, const QList<int> groups)
    {
        for (int i = 0; i < groups.size(); i++) {
            float dist = distance->compare(point, points[i]);
            if (dist < distanceThreshold)
                return groups[i];
        }
        return -1;
    }

    TemplateList templates;
};

BR_REGISTER(Transform, MeanShiftClusteringTransform)

} // namespace br

#include "cluster/meanshift.moc"
