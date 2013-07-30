#include <QApplication>
#include <QLabel>
#include <QElapsedTimer>
#include <QWaitCondition>
#include <QMutex>
#include <QMouseEvent>
#include <QPainter>

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

    QLabel *window;
    QPixmap pixmap;

    GUIProxy()
    {
        window = NULL;
    }

    bool eventFilter(QObject * obj, QEvent * event)
    {
        if (event->type() == QEvent::KeyPress)
        {
            event->accept();

            wait.wakeAll();
            return true;
        } else {
            return QObject::eventFilter(obj, event);
        }
    }

    virtual QList<QPointF> waitForKey()
    {
        QMutexLocker locker(&lock);
        wait.wait(&lock);

        return QList<QPointF>();
    }

public slots:

    void showImage(const QPixmap & input)
    {
        pixmap = input;

        window->show();
        window->setPixmap(pixmap);
        window->setFixedSize(input.size());
    }

    void createWindow()
    {
        delete window;
        QApplication::instance()->removeEventFilter(this);

        window = new QLabel();
        window->setVisible(true);

        QApplication::instance()->installEventFilter(this);
        Qt::WindowFlags flags = window->windowFlags();

        flags = flags & ~Qt::WindowCloseButtonHint;
        window->setWindowFlags(flags);
    }

};

class LandmarkProxy : public GUIProxy
{
    Q_OBJECT

public:

    bool eventFilter(QObject *obj, QEvent *event)
    {
        if (event->type() == QEvent::MouseButtonPress)
        {
            event->accept();

            QMouseEvent *mouseEvent = (QMouseEvent*)event;

            if (mouseEvent->button() == Qt::LeftButton) points.append(mouseEvent->pos());
            else if (mouseEvent->button() == Qt::RightButton && !points.isEmpty()) points.removeLast();

            QPixmap pixmapBuffer = pixmap;

            QPainter painter(&pixmapBuffer);
            painter.setBrush(Qt::red);
            foreach(const QPointF &point, points) painter.drawEllipse(point, 4, 4);

            window->setPixmap(pixmapBuffer);

            return true;
        } else {
            return GUIProxy::eventFilter(obj, event);
        }
    }

    QList<QPointF> waitForKey()
    {
        points.clear();

        GUIProxy::waitForKey();

        return points;
    }

private:

    QList<QPointF> points;
};

/*!
 * \ingroup transforms
 * \brief Displays templates in a GUI pop-up window using QT.
 * \author Charles Otto \cite caotto
 * Can be used with parallelism enabled, although it is considered TimeVarying.
 */
class ShowTransform : public TimeVaryingTransform
{
    Q_OBJECT
public:
    Q_PROPERTY(bool waitInput READ get_waitInput WRITE set_waitInput RESET reset_waitInput STORED false)
    BR_PROPERTY(bool, waitInput, true)

    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QStringList, keys, QStringList())

    ShowTransform() : TimeVaryingTransform(false, false)
    {
        gui = NULL;
        displayBuffer = NULL;
    }

    ~ShowTransform()
    {
        delete gui;
        delete displayBuffer;
    }

    void train(const TemplateList &data) { (void) data; }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        Transform * non_const = (ShowTransform *) this;
        non_const->projectUpdate(src,dst);
    }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;

        if (src.empty() || !Globals->useGui)
            return;

        foreach (const Template & t, src) {
            // build label
            QString newTitle;

            foreach (const QString & s, keys) {
                if (s.compare("name", Qt::CaseInsensitive) == 0) {
                    newTitle = newTitle + s + ": " + file.name + " ";
                } else if (t.file.contains(s)) {
                    QString out = t.file.get<QString>(s);
                    newTitle = newTitle + s + ": " + out + " ";
                }

            }
            emit this->changeTitle(newTitle);

            foreach(const cv::Mat & m, t) {
                qImageBuffer = toQImage(m);
                displayBuffer->convertFromImage(qImageBuffer);

                // Emit an explicit copy of our pixmap so that the pixmap used
                // by the main thread isn't damaged when we update displayBuffer
                // later.
                emit updateImage(displayBuffer->copy(displayBuffer->rect()));

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
        if (!Globals->useGui)
            return;

        displayBuffer = new QPixmap();

        // Create our GUI proxy
        gui = new GUIProxy();
        // Move it to the main thread, this means signals we send to it will
        // be run in the main thread, which is hopefully in an event loop
        gui->moveToThread(QApplication::instance()->thread());

        // Connect our signals to the proxy's slots
        connect(this, SIGNAL(needWindow()), gui, SLOT(createWindow()), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(updateImage(QPixmap)), gui,SLOT(showImage(QPixmap)));

        emit needWindow();
        connect(this, SIGNAL(changeTitle(QString)), gui->window, SLOT(setWindowTitle(QString)));
        connect(this, SIGNAL(hideWindow()), gui->window, SLOT(hide()));
    }

protected:
    GUIProxy * gui;
    QImage qImageBuffer;
    QPixmap * displayBuffer;

signals:
    void needWindow();
    void updateImage(const QPixmap & input);
    void changeTitle(const QString & input);
    void hideWindow();
};
BR_REGISTER(Transform, ShowTransform)

/*!
 * \ingroup transforms
 * \brief Manual selection of landmark locations
 * \author Scott Klum \cite sklum
 */
class ManualTransform : public ShowTransform
{
    Q_OBJECT

public:

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (Globals->parallelism > 1)
            qFatal("ManualTransform cannot execute in parallel.");

        dst = src;

        if (src.empty())
            return;

        for (int i = 0; i < dst.size(); i++) {
            foreach(const cv::Mat &m, dst[i]) {
                qImageBuffer = toQImage(m);
                displayBuffer->convertFromImage(qImageBuffer);

                emit updateImage(displayBuffer->copy(displayBuffer->rect()));

                // Blocking wait for a key-press
                if (this->waitInput) {
                    QList<QPointF> points = gui->waitForKey();
                    if (keys.isEmpty()) dst[i].file.appendPoints(points);
                    else {
                        if (keys.size() == points.size())
                            for (int j = 0; j < keys.size(); j++) dst[i].file.set(keys[j], points[j]);
                        else qWarning("Incorrect number of points specified for %s", qPrintable(dst[i].file.name));
                    }
                }
            }
        }
    }

    void init()
    {
        if (!Globals->useGui)
            return;

        displayBuffer = new QPixmap();

        // Create our GUI proxy
        gui = new LandmarkProxy();

        // Move it to the main thread, this means signals we send to it will
        // be run in the main thread, which is hopefully in an event loop
        gui->moveToThread(QApplication::instance()->thread());

        // Connect our signals to the proxy's slots
        connect(this, SIGNAL(needWindow()), gui, SLOT(createWindow()), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(updateImage(QPixmap)), gui,SLOT(showImage(QPixmap)));

        emit needWindow();
        connect(this, SIGNAL(hideWindow()), gui->window, SLOT(hide()));
    }
};

BR_REGISTER(Transform, ManualTransform)

class FPSLimit : public TimeVaryingTransform
{
    Q_OBJECT
    Q_PROPERTY(int targetFPS READ get_targetFPS WRITE set_targetFPS RESET reset_targetFPS STORED false)
    BR_PROPERTY(int, targetFPS, 30)

public:
    FPSLimit() : TimeVaryingTransform(false, false) {}

    ~FPSLimit() {}

    void train(const TemplateList &data) { (void) data; }


    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;
        qint64 current_time = timer.elapsed();
        qint64 target_time = last_time + target_wait;
        qint64 wait_time = target_time - current_time;

        last_time = current_time;

        if (wait_time < 0) {
            return;
        }

        QThread::msleep(wait_time);
        last_time = timer.elapsed();
    }

    void finalize(TemplateList & output)
    {
        (void) output;
    }

    void init()
    {
        target_wait = 1000.0 / targetFPS;
        timer.start();
        last_time = timer.elapsed();
    }

protected:
    QElapsedTimer timer;
    qint64 target_wait;
    qint64 last_time;
};
BR_REGISTER(Transform, FPSLimit)

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
