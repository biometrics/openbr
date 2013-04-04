#include <QFileInfo>

#include <opencv2/highgui/highgui.hpp>

#include "openbr_internal.h"

#include "core/opencvutils.h"

#include <boost/scoped_ptr.hpp>
#include <boost/filesystem.hpp>

#include "PartsBasedDetector.hpp"
#include "Candidate.hpp"
#include "FileStorageModel.hpp"
#include "Visualize.hpp"
#include "types.hpp"
#include "nms.hpp"
#include "Rect3.hpp"
#include "DistanceTransform.hpp"

using namespace cv;

namespace br
{

/*!
 * \ingroup initializers
 * \brief Initialize PBD
 * \author Scott Klum \cite sklum
 */
class PBDInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        Globals->abbreviations.insert("RectFromPBDEyes","RectFromLandmarks([9, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25],10,6.0)");
        Globals->abbreviations.insert("RectFromPBDNose","RectFromLandmarks([0, 1, 2, 3, 4, 5, 6, 7, 8],10)");
        Globals->abbreviations.insert("RectFromPBDBrow","RectFromLandmarks([15, 16, 17, 18, 19, 26, 27, 28, 29, 30],10)");
        Globals->abbreviations.insert("RectFromPBDMouth","RectFromLandmarks([31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],10)");
        Globals->abbreviations.insert("RectFromPBDJaw","RectFromLandmarks([51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],10)");
    }

    void finalize() const
    {

    }
};

BR_REGISTER(Initializer, PBDInitializer)

/*!
 * \ingroup transforms
 * \brief Wraps Parts Based Detector
 * \author Scott Klum \cite sklum
 * \note Some of the detected landmarks overlap
 * \todo Remove print statements from pdb src, remove Boost dependency
 */

class PBDTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QString modelPath READ get_modelPath WRITE set_modelPath RESET reset_modelPath STORED false)
    BR_PROPERTY(QString, modelPath, "")

    boost::scoped_ptr<Model> model;

    void init()
    {
        if (modelPath.isEmpty()) qFatal("No model file");

        model.reset(new FileStorageModel);

        QFileInfo info(modelPath);

        if (info.suffix().compare("xml") == 0 || info.suffix().compare("yaml") == 0) model.reset(new FileStorageModel);
        else qFatal("Unsupported model format: %s", qPrintable(info.suffix()));

        if (!(model->deserialize(info.filePath().toStdString()))) qFatal("Error deserializing model, check file path");
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src.m().clone();

        PartsBasedDetector<float> pbd;
        pbd.distributeModel(*model);

        Mat_<float> depth;

        // Detect potential candidates in the image
        vector<Candidate> candidates;
        pbd.detect(src.m(), depth, candidates);

        if (Globals->verbose) qDebug("%li candidate detected for %s", candidates.size(), qPrintable(src.file.flat()));

        // Sort the candidates by score, then supress all but the candidate with the maximum score
        if (candidates.size() > 0) {
            Candidate::sort(candidates);
            Candidate::nonMaximaSuppression(src.m(), candidates, 0.2);
            foreach ( const cv::Rect part, candidates[0].parts() )
                dst.file.appendLandmark(QPointF(part.x + part.width/2.0, part.y + part.height/2.0));
        }
    }
};

BR_REGISTER(Transform, PBDTransform)

} // namespace br

#include "pbd.moc"

