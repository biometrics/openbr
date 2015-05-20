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

#include <fstream>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/qtutils.h>

#include <opencv2/highgui/highgui.hpp>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Applies a classifier to a sliding window.
 * \author Jordan Cheney \cite JordanCheney
 */

class SlidingWindowTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(br::Classifier *classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    Q_PROPERTY(QString cascadeDir READ get_cascadeDir WRITE set_cascadeDir RESET reset_cascadeDir STORED false)
    BR_PROPERTY(br::Classifier *, classifier, NULL)
    BR_PROPERTY(QString, cascadeDir, "")

    void train(const TemplateList &data)
    {
        classifier->train(data.data(), File::get<float>(data, "Label", -1));
    }

    void project(const Template &src, Template &dst) const
    {
        (void)src; (void)dst;
    }

    void load(QDataStream &stream)
    {
        (void) stream;
        return;
    }

    void store(QDataStream &stream) const
    {
        (void) stream;

	QString path = Globals->sdkPath + "/share/openbr/models/openbrcascades/" + cascadeDir;
	QtUtils::touchDir(QDir(path));

	QString filename = path + "/cascade.xml";
        FileStorage fs(filename.toStdString(), FileStorage::WRITE);

        if (!fs.isOpened()) {
	    qWarning("Unable to open file: %s", qPrintable(filename));
            return;
	}
	
        fs << FileStorage::getDefaultObjectName(filename.toStdString()) << "{";

        classifier->write(fs);

        fs << "}";
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

} // namespace br

#include "imgproc/slidingwindow.moc"
