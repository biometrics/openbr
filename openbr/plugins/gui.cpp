#include <QApplication>
#include <QLabel>
#include <QElapsedTimer>
#include <QWaitCondition>
#include <QMutex>
#include <opencv2/imgproc/imgproc.hpp>
#include "openbr_internal.h"

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
    QMutex lock;
    QWaitCondition wait;

public:

    bool eventFilter(QObject * obj, QEvent * event)
    {
        if (event->type() == QEvent::KeyPress)
        {
            wait.wakeAll();
        }
        return QObject::eventFilter(obj, event);
    }

    QLabel * window;
    GUIProxy()
    {
        window = NULL;
    }

    void waitForKey()
    {
        QMutexLocker locker(&lock);
        wait.wait(&lock);
    }

public slots:

    void showImage(const QPixmap & input)
    {
        window->show();
        window->setPixmap(input);
        window->setFixedSize(input.size());
    }

    void createWindow()
    {
        delete window;
        window = new QLabel();
        window->setVisible(true);

        QApplication::instance()->installEventFilter(this);
        Qt::WindowFlags flags = window->windowFlags();

        flags = flags & ~Qt::WindowCloseButtonHint;
        window->setWindowFlags(flags);
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
    Q_PROPERTY(bool waitInput READ get_waitInput WRITE set_waitInput RESET reset_waitInput STORED false)
    BR_PROPERTY(bool, waitInput, false)

    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QStringList, keys, QStringList("FrameNumber"))


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
            // build label
            QString newTitle;
            foreach (const QString & s, keys) {
                if (t.file.contains(s)) {
                    QString out = t.file.get<QString>(s);
                    newTitle = newTitle + s + ": " + out + " ";
                }

            }
            emit this->changeTitle(newTitle);

            foreach(const cv::Mat & m, t) {
                qImageBuffer = toQImage(m);
                displayBuffer.convertFromImage(qImageBuffer);

                // Emit an explicit  copy of our pixmap so that the pixmap used
                // by the main thread isn't damaged when we update displayBuffer
                // later.
                emit updateImage(displayBuffer.copy(displayBuffer.rect()));

                // Blocking wait for a key-press
                if (this->waitInput)
                    gui->waitForKey();

            }
        }
    }

    void finalize(TemplateList & output)
    {
        (void) output;
        emit hideWindow();
    }

    void init()
    {
        emit needWindow();
        connect(this, SIGNAL(changeTitle(QString)), gui->window, SLOT(setWindowTitle(QString)));
        connect(this, SIGNAL(hideWindow()), gui->window, SLOT(hide()));
    }

protected:
    GUIProxy * gui;
    QImage qImageBuffer;
    QPixmap displayBuffer;

signals:
    void needWindow();
    void updateImage(const QPixmap & input);
    void changeTitle(const QString & input);
    void hideWindow();
};

BR_REGISTER(Transform, Show2Transform)

class FPSSynch : public TimeVaryingTransform
{
    Q_OBJECT
    Q_PROPERTY(int targetFPS READ get_targetFPS WRITE set_targetFPS RESET reset_targetFPS STORED false)
    BR_PROPERTY(int, targetFPS, 30)

public:
    FPSSynch() : TimeVaryingTransform(false, false) {}

    ~FPSSynch() {}

    void train(const TemplateList &data) { (void) data; }


    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;
        qint64 time_delta = timer.elapsed();

        qint64 wait_time = target_wait - time_delta;
        timer.start();

        if (wait_time < 0) {
            return;
        }

        QThread::msleep(wait_time);
    }

    void finalize(TemplateList & output)
    {
        (void) output;
    }

    void init()
    {
        target_wait = 1000 / targetFPS;
        timer.start();
    }

protected:
    QElapsedTimer timer;
    qint64 target_wait;
};
BR_REGISTER(Transform, FPSSynch)

class FPSCalc : public TimeVaryingTransform
{
    Q_OBJECT
    Q_PROPERTY(int targetFPS READ get_targetFPS WRITE set_targetFPS RESET reset_targetFPS)
    BR_PROPERTY(int, targetFPS, 30)


public:
    FPSCalc() : TimeVaryingTransform(false, false) { initialized = false; }

    ~FPSCalc() {}

    void train(const TemplateList &data) { (void) data; }


    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;

        if (!initialized) {
            initialized = true;
            timer.start();
        }
        framesSeen++;

        if (dst.empty())
            return;

        qint64 elapsed = timer.elapsed();
        if (elapsed > 1000) {
            double fps = 1000 * framesSeen / elapsed;
            dst.first().file.set("AvgFPS", fps);
        }
    }

    void finalize(TemplateList & output)
    {
        (void) output;
    }

    void init()
    {
        initialized = false;
        framesSeen = 0;
    }

protected:
    bool initialized;
    QElapsedTimer timer;
    qint64 framesSeen;
};
BR_REGISTER(Transform, FPSCalc)

} // namespace br

#include "gui.moc"
