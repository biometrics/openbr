#include <QFileInfo>

#include <opencv2/highgui/highgui.hpp>

#include <openbr_plugin.h>

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
 * \ingroup transforms
 * \brief Wraps Parts Based Detector
 * \author Scott Klum \cite sklum
 * \todo Remove print statements from pdb src
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

        // Sort the candidates by score, then supress all but the candidate with the maximum score
        if (candidates.size() > 0) {
            Candidate::sort(candidates);
            Candidate::nonMaximaSuppression(src.m(), candidates, 0.2);
        }

        foreach ( const cv::Rect part, candidates[0].parts() )
            dst.file.appendLandmark(QPointF(part.x + part.width/2.0, part.y + part.height/2.0));
    }
};

BR_REGISTER(Transform, PBDTransform)

} // namespace br

#include "pbd.moc"

