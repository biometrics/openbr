#include <QApplication>
#include <QLabel>
#include <opencv2/imgproc/imgproc.hpp>
#include <openbr/openbr_plugin.h>

using namespace cv;

namespace br
{
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
public:
    QLabel * window;

    GUIProxy()
    {
        window = NULL;
    }

public slots:

    void showImage(const QPixmap & input)
    {
        window->setPixmap(input);
        window->setFixedSize(input.size());
        window->setVisible(true);
    }

    void createWindow()
    {
        delete window;
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
        connect(this, SIGNAL(updateImage(QPixmap)), gui,SLOT(showImage(QPixmap)));
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
                qImageBuffer = toQImage(m);
                displayBuffer.convertFromImage(qImageBuffer);
                // Emit an explicit  copy of our pixmap so that the pixmap used
                // by the main thread isn't damaged when we update displayBuffer
                // later.
                emit updateImage(displayBuffer.copy(displayBuffer.rect()));
            }
        }
    }

    void finalize(TemplateList & output)
    {
        (void) output;
        // todo: hide window?
    }

    void init()
    {
        emit needWindow();
    }

protected:
    GUIProxy * gui;
    QImage qImageBuffer;
    QPixmap displayBuffer;

signals:
    void needWindow();
    void updateImage(const QPixmap & input);
};

BR_REGISTER(Transform, Show2Transform)

} // namespace br

#include "gui.moc"
