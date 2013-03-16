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

#ifndef BR_EMBEDDED
#include <QLabel>
#include <opencv2/imgproc/imgproc.hpp>
#include <QApplication>
#endif


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
            if (t.isEmpty()) qWarning("Can't open %s from %s", qPrintable(file.flat()), qPrintable(QDir::currentPath()));
            dst.append(t);
            dst.file.append(t.file.localMetadata());
        }
        dst.file.set("FTO", dst.isEmpty());
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

#ifndef BR_EMBEDDED

QImage toQImage(const Mat &mat)
{
    // Convert to 8U depth
    Mat mat8u;
    if (mat.depth() != CV_8U) {
        double globalMin = std::numeric_limits<double>::max();
        double globalMax = -std::numeric_limits<double>::max();

        std::vector<Mat> mv;
        split(mat, mv);
        for (size_t i=0; i<mv.size(); i++) {
            double min, max;
            minMaxLoc(mv[i], &min, &max);
            globalMin = std::min(globalMin, min);
            globalMax = std::max(globalMax, max);
        }
        assert(globalMax >= globalMin);

        double range = globalMax - globalMin;
        if (range != 0) {
            double scale = 255 / range;
            convertScaleAbs(mat, mat8u, scale, -(globalMin * scale));
        } else {
            // Monochromatic
            mat8u = Mat(mat.size(), CV_8UC1, Scalar((globalMin+globalMax)/2));
        }
    } else {
        mat8u = mat;
    }

    // Convert to 3 channels
    Mat mat8uc3;
    if      (mat8u.channels() == 4) cvtColor(mat8u, mat8uc3, CV_BGRA2RGB);
    else if (mat8u.channels() == 3) cvtColor(mat8u, mat8uc3, CV_BGR2RGB);
    else if (mat8u.channels() == 1) cvtColor(mat8u, mat8uc3, CV_GRAY2RGB);

    return QImage(mat8uc3.data, mat8uc3.cols, mat8uc3.rows, 3*mat8uc3.cols, QImage::Format_RGB888).copy();
}

// Provides slots for manipulating a QLabel, but does not inherit from QWidget.
// Therefore, it can be moved to the main thread if not created there initially
// since god forbid you create a QWidget subclass in not the main thread.
class GUIProxy : public QObject
{
    Q_OBJECT

    QLabel * window;

public:
    GUIProxy()
    {
        window =NULL;
    }

public slots:
    void showImage(QPixmap pixmap)
    {
        window->show();
        window->setPixmap(pixmap);
        window->setFixedSize(pixmap.size());
        window->update();
    }

    void createWindow()
    {
        delete window;
        window = NULL;
        window = new QLabel();
    }
};

/*!
 * \ingroup transforms
 * \brief Displays templates in a GUI pop-up window using QT.
 * \author Charles Otto \cite caotto
 * Unlike ShowTransform, this can be used with parallelism enabled, although it
 * is considered TimeVarying.
 */
class Show2Transform : public TimeVaryingTransform
{
    Q_OBJECT
public:
    Show2Transform() : TimeVaryingTransform(false, false)
    {
        // Create our GUI proxy
        gui = new GUIProxy();
        // Move it to the main thread, this means signals we send to it will
        // be run in the main thread, which is hopefully in an event loop
        gui->moveToThread(QApplication::instance()->thread());
        // Connect our signals to the proxy's slots
        connect(this, SIGNAL(needWindow()), gui, SLOT(createWindow()), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(updateImage(QPixmap)), gui, SLOT(showImage(QPixmap)));
    }

    ~Show2Transform()
    {
        delete gui;
    }

    void train(const TemplateList &data) { (void) data; }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        Transform * non_const = (Show2Transform *) this;
        non_const->projectUpdate(src,dst);
    }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;

        if (src.empty())
            return;

        foreach (const Template & t, src) {
            foreach(const cv::Mat & m, t) {
                QImage qImageBuffer = toQImage(m);
                displayBuffer.convertFromImage(qImageBuffer);
                emit updateImage(displayBuffer);
            }
        }
    }

    void finalize(TemplateList & output)
    {
        (void) output;
        // todo: hide ui?
    }

    void init()
    {
        emit needWindow();
    }

protected:
    GUIProxy * gui;
    QPixmap displayBuffer;

signals:
    void needWindow();
    void updateImage(QPixmap input);
};

BR_REGISTER(Transform, Show2Transform)
#endif

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
    Q_PROPERTY(bool nTemplates READ get_data WRITE set_data RESET reset_data)
    BR_PROPERTY(bool, error, true)
    BR_PROPERTY(bool, data, false)
    BR_PROPERTY(bool, size, false)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        const QString nameString = src.file.flat();
        const QString dataString = data ? OpenCVUtils::matrixToString(src)+"\n" : QString();
        const QString nTemplates = size ? QString::number(src.size()) : QString();
        if (error) qDebug("%s\n%s\n%s", qPrintable(nameString), qPrintable(dataString), qPrintable(nTemplates));
        else       printf("%s\n%s\n%s", qPrintable(nameString), qPrintable(dataString), qPrintable(nTemplates));
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
    static int count;
    int index;

 public:
    CheckTransform() : index(count++) {}

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
                    qFatal("%s NaN check %d failed!", qPrintable(src.file.flat()), index);
        }
    }
};

int CheckTransform::count = 0;

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
 * \brief Rename metadata key
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
        if (dst.file.localKeys().contains(find)) {
            dst.file.set(replace, dst.file.value(find));
            dst.file.remove(find);
        }
    }
};

BR_REGISTER(Transform, RenameTransform)

/*!
 * \ingroup transforms
 * \brief Rename first found metadata key
 * \author Josh Klontz \cite jklontz
 */
class RenameFirstTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList find READ get_find WRITE set_find RESET reset_find STORED false)
    Q_PROPERTY(QString replace READ get_replace WRITE set_replace RESET reset_replace STORED false)
    BR_PROPERTY(QStringList, find, QStringList())
    BR_PROPERTY(QString, replace, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        foreach (const QString &key, find)
            if (dst.file.localKeys().contains(key)) {
                dst.file.set(replace, dst.file.value(key));
                dst.file.remove(key);
                break;
            }
    }
};

BR_REGISTER(Transform, RenameFirstTransform)

/*!
 * \ingroup transforms
 * \brief Change the br::Template::file extension
 * \author Josh Klontz \cite jklontz
 */
class AsTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString extension READ get_extension WRITE set_extension RESET reset_extension STORED false)
    BR_PROPERTY(QString, extension, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.name = dst.file.name.left(dst.file.name.lastIndexOf('.')+1) + extension;
    }
};

BR_REGISTER(Transform, AsTransform)

}

#include "misc.moc"
