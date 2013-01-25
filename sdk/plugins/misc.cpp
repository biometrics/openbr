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

#include <opencv2/highgui/highgui.hpp>
#include <openbr_plugin.h>

#include "core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Applies br::Format to br::Template::file::name and appends results.
 * \author Josh Klontz \cite jklontz
 */
class OpenTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (Globals->verbose) qDebug("Opening %s", qPrintable(src.file.flat()));
        dst.file = src.file;
        foreach (const File &file, src.file.split()) {
            QScopedPointer<Format> format(Factory<Format>::make(file));
            Template t = format->read();
            if (t.isEmpty()) qWarning("Can't open %s", qPrintable(file.flat()));
            dst.append(t);
            dst.file.append(t.file.localMetadata());
        }
        dst.file.insert("FTO", dst.isEmpty());
    }
};

BR_REGISTER(Transform, OpenTransform)

/*!
 * \ingroup transforms
 * \brief Displays templates in a GUI pop-up window.
 * \author Josh Klontz \cite jklontz
 */
class ShowTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(bool waitKey READ get_waitKey WRITE set_waitKey RESET reset_waitKey STORED false)
    BR_PROPERTY(bool, waitKey, true)

    static int counter;
    int uid;

    void init()
    {
        uid = counter++;
        Globals->setProperty("parallelism", "0"); // Can only work in single threaded mode
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        if (Globals->parallelism) {
            qWarning("Show::project() only works in single threaded mode.");
            return;
        }

        for (int i=0; i<src.size(); i++)
            OpenCVUtils::showImage(src[i], "Show" + (counter*src.size() > 1 ? "-" + QString::number(uid*src.size()+i) : QString()), false);

        if (waitKey && !src.isEmpty()) cv::waitKey(-1);
    }
};

int ShowTransform::counter = 0;

BR_REGISTER(Transform, ShowTransform)

/*!
 * \ingroup transforms
 * \brief Prints the template's file to stdout or stderr.
 * \author Josh Klontz \cite jklontz
 */
class PrintTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(bool error READ get_error WRITE set_error RESET reset_error)
    Q_PROPERTY(bool data READ get_data WRITE set_data RESET reset_data)
    BR_PROPERTY(bool, error, false)
    BR_PROPERTY(bool, data, false)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        const QString nameString = src.file.flat();
        const QString dataString = data ? OpenCVUtils::matrixToString(src)+"\n" : QString();
        if (error) qDebug("%s\n%s", qPrintable(nameString), qPrintable(dataString));
        else       printf("%s\n%s", qPrintable(nameString), qPrintable(dataString));
    }
};

BR_REGISTER(Transform, PrintTransform)

/*!
 * \ingroup transforms
 * \brief Checks the template for NaN values.
 * \author Josh Klontz \cite jklontz
 */
class CheckTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        foreach (const Mat &m, src) {
            Mat fm;
            m.convertTo(fm, CV_32F);
            const int elements = fm.rows * fm.cols * fm.channels();
            const float *data = (const float*)fm.data;
            for (int i=0; i<elements; i++)
                if (data[i] != data[i])
                    qFatal("%s NaN check failed!", qPrintable(src.file.flat()));
        }
    }
};

BR_REGISTER(Transform, CheckTransform)

/*!
 * \ingroup transforms
 * \brief Sets the template's matrix data to the br::File::name.
 * \author Josh Klontz \cite jklontz
 */
class NameTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        QByteArray name = src.file.baseName().toLocal8Bit();
        dst = Mat(1, name.size(), CV_8UC1);
        memcpy(dst.m().data, name.data(), name.size());
    }
};

BR_REGISTER(Transform, NameTransform)

/*!
 * \ingroup transforms
 * \brief A no-op transform.
 * \see DiscardTransform FirstTransform RestTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class IdentityTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;
    }
};

BR_REGISTER(Transform, IdentityTransform)

/*!
 * \ingroup transforms
 * \brief Removes all template's matrices.
 * \see IdentityTransform FirstTransform RestTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class DiscardTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
    }
};

BR_REGISTER(Transform, DiscardTransform)

/*!
 * \ingroup transforms
 * \brief Removes all but the first matrix from the template.
 * \see IdentityTransform DiscardTransform RestTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class FirstTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        dst = src.m();
    }
};

BR_REGISTER(Transform, FirstTransform)

/*!
 * \ingroup transforms
 * \brief Removes the first matrix from the template.
 * \see IdentityTransform DiscardTransform FirstTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class RestTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.removeFirst();
    }
};

BR_REGISTER(Transform, RestTransform)

/*!
 * \ingroup transforms
 * \brief Removes the matrix from the template at the specified index.
 * \author Josh Klontz \cite jklontz
 * \see IdentityTransform DiscardTransform FirstTransform RestTransform
 */
//! [example_transform]
class RemoveTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int index READ get_index WRITE set_index RESET reset_index STORED false)
    BR_PROPERTY(int, index, 0)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.removeAt(index);
    }
};

BR_REGISTER(Transform, RemoveTransform)
//! [example_transform]

/*!
 * \ingroup transforms
 * \brief Rename metadata
 * \author Josh Klontz \cite jklontz
 */
class RenameTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString find READ get_find WRITE set_find RESET reset_find STORED false)
    Q_PROPERTY(QString replace READ get_replace WRITE set_replace RESET reset_replace STORED false)
    BR_PROPERTY(QString, find, "")
    BR_PROPERTY(QString, replace, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        foreach (const QString &key, dst.file.localKeys())
            if (key.contains(find)) {
                QString newKey = QString(key).replace(find, replace);
                dst.file.insert(newKey, dst.file.get(key));
                dst.file.remove(key);
            }
    }
};

BR_REGISTER(Transform, RenameTransform)

}

#include "misc.moc"
