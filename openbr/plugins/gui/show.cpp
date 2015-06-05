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

#include <QApplication>
#include <QLabel>
#include <QElapsedTimer>
#include <QInputDialog>
#include <QWaitCondition>
#include <QMutex>
#include <QMouseEvent>
#include <QPainter>
#include <QMainWindow>
#include <QPushButton>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QLineEdit>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/gui/utility.h>

using namespace cv;

namespace br
{
// Generally speaking, Qt wants GUI objects to be on the main thread, and
// for the main thread to be in an event loop. We don't restrict transform
// creation to just the main thread, but in br we do compromise and put
// the main thread in an event loop (and so should any applications wanting to
// use GUI transforms). This does mean we need a way to make our QWidget subclasses
// on the main thread. We can't create them from an arbitrary thread, then move them
// (you know, since that would be crazy), so we need some tricks to get the main
// thread to make these objects.

// Part 1. A generic interface for creating objects, the type of object
// created is not exposed in the interface.
class NominalCreation
{
public:
    virtual ~NominalCreation() {}
    virtual void creation()=0;
};

// Part 2. A template class that creates an object of the specified type
// through the interface defined in part 1. The point of this is that the
// type of object created can be hidden by using a NominalCreation *.
template<typename T>
class ActualCreation : public NominalCreation
{
public:
    T *basis;

    void creation()
    {
        basis = new T();
    }
};

// Part 3. A class that inherits from QObject, but not QWidget. This means
// we are free to move it to the main thread.
// If this object is on the main thread, and we signal one of its slots, then
// the slot will be executed by the main thread, which is what we need.
// Unfortunately, since it uses Q_OBJECT we cannot make it a template class, but
// we still want to be able to make objects of arbitrary type on the main thread,
// so that we don't need a different adaptor for every type of QWidget subclass we use.
class MainThreadCreator : public QObject
{
    Q_OBJECT
public:

    MainThreadCreator()
    {
        this->moveToThread(QApplication::instance()->thread());
        // We actually bind a signal on this object to one of its own slots.
        // the signal will be emitted by a call to getItem from an abitrary
        // thread.
        connect(this, SIGNAL(needCreation()), this, SLOT(createThing()), Qt::BlockingQueuedConnection);
    }

    // While this cannot be a template class, it can still have a template
    // method. Which is useful, but the slot which will actually be executed
    // by the main thread cannot be a template. So, we use the template method
    // here (called from an arbitrary thread, to create an object of arbitrary type)
    // to instantiate an ActualCreation object with matching type, then we hide
    // the template with the NominalCreation interface, and call worker->creation
    // in the slot.
    template<typename T>
    T *getItem()
    {
        // If this is called by the main thread, we can just create the object
        // it's important to check, otherwise we will have problems trying to
        // wait for a blocking connection that is supposed to be processed by
        // the thread that is waiting.
        if (QThread::currentThread() == QApplication::instance()->thread())
            return new T();

        // Create the object creation interface
        ActualCreation<T> * actualWorker;
        actualWorker = new ActualCreation<T> ();
        // hide it
        worker = actualWorker;

        // emit the signal, we set up a blocking queued connection, so
        // this is a blocking wait for the slot to finish being run.
        emit needCreation();

        // collect the results, and return.
        T *output = actualWorker->basis;
        delete actualWorker;
        return output;
    }

    NominalCreation *worker;

signals:
    void needCreation();

public slots:
    // The actual slot, to be run by the main thread. The type
    // of object being created is not, and indeed cannot, be exposed here
    // since this cannot be a template method, and the class cannot be a
    // template class.
    void createThing()
    {
        worker->creation();
    }
};

class DisplayWindow : public QLabel
{
    Q_OBJECT

protected:
    QMutex lock;
    QWaitCondition wait;
    QPixmap pixmap;

public:

    DisplayWindow(QWidget *parent = NULL) : QLabel(parent)
    {
        setFixedSize(200,200);
        QApplication::instance()->installEventFilter(this);
    }
    ~DisplayWindow()
    {
        QApplication::instance()->removeEventFilter(this);
    }

public slots:
    void showImage(const QPixmap &input)
    {
        pixmap = input;

        show();
        setPixmap(pixmap);

        // We appear to get a warning on windows if we set window width < 104. This is of course not
        // reflected in the Qt min size settings, and I don't know how to query it.
        QSize temp = input.size();
        if (temp.width() < 104)
            temp.setWidth(104);
        setFixedSize(temp);
    }

    bool eventFilter(QObject *obj, QEvent *event)
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
};

class RectMarkingWindow : public DisplayWindow
{
public:
    RectMarkingWindow() : DisplayWindow()
    {
        drawingRect = false;
    }

    bool drawingRect;
    QVector<QRectF> rects;
    QList<QString> rectLabels;

    QPointF rectOrigin;
    QPointF currentEnd;
    QRectF currentRect;
    bool disableAccept;

    bool eventFilter(QObject *obj, QEvent *event)
    {
        if (disableAccept)
            return QObject::eventFilter(obj, event);


        if (event->type() == QEvent::MouseButtonPress || event->type() == QEvent::MouseMove)
        {
            event->accept();

            QMouseEvent *mouseEvent = (QMouseEvent*)event;

            if (event->type() == QEvent::MouseButtonPress)
            {

                if (mouseEvent->button() == Qt::LeftButton) {
                    if (!drawingRect)
                    {
                        drawingRect = true;
                        rectOrigin = mouseEvent->pos();
                        return true;
                    }
                    else
                    {
                        drawingRect = false;

                        rects.append(QRectF(rectOrigin, mouseEvent->pos()));
                        rects.last() = rects.last().normalized();
                        // If no labels were provided, we store everything as anonymous rectangles
                        if (promptKeys.empty())
                           rectLabels.append("rects");
                        // otherwise, prompt the user to select a label
                        else
                        {
                            // get a label from the user
                            bool ok = false;

                            // Don't intercept events while the sub-dialog is up (if we take the events, then it will not work correctly)
                            disableAccept = true;
                            QString res = QInputDialog::getItem(this, "Select a label", "", promptKeys, next_idx, false, &ok);

                            disableAccept = false;
                            if (ok) {
                                rectLabels.append(res);
                                for (int i=0; i < promptKeys.size(); i++)
                                {
                                    if (res == promptKeys[i]) {
                                        next_idx = (i + 1) % promptKeys.size();
                                        break;
                                    }
                                }
                            }
                            else {
                                rects.remove(rects.size()-1);
                            }
                        }
                    }
                }
                // rclick -- reset state if drawing, remove last rect if done
                else if (mouseEvent->button() == Qt::RightButton && (!rects.isEmpty() || drawingRect))
                {
                    if (drawingRect)
                        drawingRect = false;
                    else
                    {
                        rects.remove(rects.size()-1);
                        rectLabels.removeLast();
                    }
                }
            }
            else
                currentEnd = mouseEvent->pos();
            QPixmap pixmapBuffer = pixmap;

            QPainter painter(&pixmapBuffer);
            painter.setPen(Qt::red);

            painter.drawRects(rects);

            if (drawingRect)
            {
                currentRect = QRectF(rectOrigin, currentEnd);
                painter.setPen(Qt::green);
                painter.drawRect(currentRect);
            }

            setPixmap(pixmapBuffer);

            return true;
        } else {
            if (event->type() == QEvent::KeyPress)
            {
                QKeyEvent *kevent = (QKeyEvent *) event;
                if (kevent->key() == Qt::Key_Enter || kevent->key() == Qt::Key_Return) {
                    event->accept();
                    return true;
                }
            }
            return DisplayWindow::eventFilter(obj, event);
        }
    }


    QList<QPointF> waitForKey()
    {

        rects.clear();
        drawingRect = false;
        disableAccept = false;
        next_idx = 0;
        DisplayWindow::waitForKey();

        return QList<QPointF>();
    }

    void setKeys(const QStringList &keys)
    {
        promptKeys = keys;
    }

    int next_idx;
private:
    QStringList promptKeys;

};

class PointMarkingWindow : public DisplayWindow
{
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
            foreach (const QPointF &point, points) painter.drawEllipse(point, 4, 4);

            setPixmap(pixmapBuffer);

            return true;
        } else {
            return DisplayWindow::eventFilter(obj, event);
        }
    }

    QList<QPointF> waitForKey()
    {
        points.clear();

        DisplayWindow::waitForKey();

        return points;
    }

private:
    QList<QPointF> points;


};

class PromptWindow : public DisplayWindow
{
    bool eventFilter(QObject *obj, QEvent *event)
    {
        if (event->type() == QEvent::KeyPress)
        {
            event->accept();

            QKeyEvent *key_event = dynamic_cast<QKeyEvent *> (event);
            if (key_event == NULL) {
                qDebug("failed to donwcast key event");
                return true;
            }

            QString text = key_event->text();

            text =text.toLower();
            if (text == "y" || text == "n")
            {
                gotString = key_event->text();
                wait.wakeAll();
            }
            else qDebug("Please answer y/n");

            return true;
        } else {
            return QObject::eventFilter(obj, event);
        }
    }

public:
    QString waitForKeyPress()
    {
        QMutexLocker locker(&lock);
        wait.wait(&lock);

        return gotString;
    }

private:
    QString gotString;


};

class GUIWindow : public QMainWindow
{
    Q_OBJECT

public:

    GUIWindow(QWidget *parent = NULL) : QMainWindow(parent)
    {
        centralWidget = new QWidget();
        layout = new QHBoxLayout();
        inputLayout = new QVBoxLayout();

        button.setText("Set Template Metadata");

        layout->addWidget(&label);

        inputLayout->addWidget(&button);
        layout->addLayout(inputLayout);

        centralWidget->setLayout(layout);

        setCentralWidget(centralWidget);

        connect(&button, SIGNAL(clicked()), this, SLOT(buttonPressed()));
    }

public slots:
    void showImage(const QPixmap &input)
    {
        hide();

        createForm();
        pixmap = input;
        label.setPixmap(pixmap);
        label.setFixedSize(input.size());

        show();
    }

    QStringList waitForButtonPress()
    {
        QMutexLocker locker(&lock);
        wait.wait(&lock);

        QStringList values;
        for (int i = 0; i<fields.size(); i++) {
            values.append(fields.at(i)->text());
            fields[i]->clear();
        }
        return values;
    }

public slots:

    void buttonPressed()
    {
        wait.wakeAll();
    }

    void setKeys(const QStringList& k)
    {
        keys = k;
    }

private:

    void createForm()
    {
        if (fields.size() != keys.size()) {
            foreach (const QString& label, keys) {
                QLineEdit *edit = new QLineEdit;
                fields.append(edit);
                QFormLayout *form = new QFormLayout;
                form->addRow(label, edit);
                inputLayout->addLayout(form);
            }
        }
    }

    QWidget *centralWidget;
    QStringList keys;
    QList<QLineEdit*> fields;
    QPushButton button;
    QMutex lock;
    QWaitCondition wait;
    QPixmap pixmap;
    QLabel label;
    QHBoxLayout *layout;
    QVBoxLayout *inputLayout;

};

/*!
 * \ingroup transforms
 * \brief Displays templates in a GUI pop-up window using QT.
 *
 * Can be used with parallelism enabled, although it is considered TimeVarying.
 *
 * \author Charles Otto \cite caotto
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
        displayBuffer = NULL;
        window = NULL;
    }

    ~ShowTransform()
    {
        delete displayBuffer;
        if (QThread::currentThread() == QCoreApplication::instance()->thread())
            delete window;
        else
            emit destroyWindow();
    }

    void train(const TemplateList &data) { (void) data; }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;

        if (src.empty())
            return;

        foreach (const Template &t, src) {
            // build label
            QString newTitle;

            foreach (const QString &s, keys) {
                if (s.compare("name", Qt::CaseInsensitive) == 0) {
                    newTitle = newTitle + s + ": " + t.file.fileName() + " ";
                } else if (t.file.contains(s)) {
                    QString out = t.file.get<QString>(s);
                    newTitle = newTitle + s + ": " + out + " ";
                }

            }
            emit this->changeTitle(newTitle);

            foreach (const cv::Mat &m, t) {
                if (!m.data) continue;
                qImageBuffer = toQImage(m);
                displayBuffer->convertFromImage(qImageBuffer);

                // Emit an explicit copy of our pixmap so that the pixmap used
                // by the main thread isn't damaged when we update displayBuffer
                // later.
                emit updateImage(displayBuffer->copy(displayBuffer->rect()));

                // Blocking wait for a key-press
                if (this->waitInput)
                    window->waitForKey();

            }
        }
    }

    void finalize(TemplateList &output)
    {
        (void) output;
        emit hideWindow();
    }

    void init()
    {
        initActual<DisplayWindow>();
    }

    template<typename WindowType>
    void initActual()
    {
        if (!Globals->useGui) {
            qWarning("GUI transform %s created without enabling GUI support.\nRun \"br -gui ...\" to enable GUI support from the command line, or set\nGlobals->useGui to true.", this->metaObject()->className());
            return;
        }

        if (displayBuffer)
            delete displayBuffer;
        displayBuffer = new QPixmap();

        if (window)
            window->deleteLater();

        window = creator.getItem<WindowType>();
        // Connect our signals to the window's slots
        connect(this, SIGNAL(updateImage(QPixmap)), window,SLOT(showImage(QPixmap)));
        connect(this, SIGNAL(changeTitle(QString)), window, SLOT(setWindowTitle(QString)));
        connect(this, SIGNAL(hideWindow()), window, SLOT(hide()));
        connect(this, SIGNAL(destroyWindow()), window, SLOT(deleteLater()), Qt::BlockingQueuedConnection);
    }

protected:
    MainThreadCreator creator;
    DisplayWindow *window;
    QImage qImageBuffer;
    QPixmap *displayBuffer;

signals:
    void updateImage(const QPixmap &input);
    void changeTitle(const QString &input);
    void hideWindow();
    void destroyWindow();
};
BR_REGISTER(Transform, ShowTransform)

/*!
 * \ingroup transforms
 * \brief Show the training data
 * \author Josh Klontz \cite jklontz
 */
class ShowTrainingTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform *show READ get_show WRITE set_show RESET reset_show STORED false)
    BR_PROPERTY(br::Transform*, show, make("Show"))

    void train(const TemplateList &data)
    {
        TemplateList dst;
        show->project(data, dst);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
    }
};
BR_REGISTER(Transform, ShowTrainingTransform)

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
            foreach (const cv::Mat &m, dst[i]) {
                qImageBuffer = toQImage(m);
                displayBuffer->convertFromImage(qImageBuffer);

                emit updateImage(displayBuffer->copy(displayBuffer->rect()));

                // Blocking wait for a key-press
                if (this->waitInput) {
                    QList<QPointF> points = window->waitForKey();
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
        initActual<PointMarkingWindow>();
    }
};

BR_REGISTER(Transform, ManualTransform)

/*!
 * \ingroup transforms
 * \brief Manual select rectangular regions on an image.
 * Stores marked rectangles as anonymous rectangles, or if a set of labels is provided, prompt the user
 * to select one of those labels after drawing each rectangle.
 * \author Charles Otto \cite caotto
 */
class ManualRectsTransform : public ShowTransform
{
    Q_OBJECT

public:

    Q_PROPERTY(QStringList labels READ get_labels WRITE set_labels RESET reset_labels STORED false)
    BR_PROPERTY(QStringList, labels, QStringList())

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {

        dst = src;

        if (!Globals->useGui)
            return;
        if (src.empty())
            return;

        for (int i = 0; i < dst.size(); i++) {
            foreach (const cv::Mat &m, dst[i]) {
                qImageBuffer = toQImage(m);
                displayBuffer->convertFromImage(qImageBuffer);

                emit updateImage(displayBuffer->copy(displayBuffer->rect()));

                // Blocking wait for a key-press
                if (this->waitInput) {
                    window->waitForKey();
                    QVector<QRectF> rectSet  = trueWindow->rects;
                    QList<QString> labelSet= trueWindow->rectLabels;

                    for (int idx = 0; idx < rectSet.size(); idx++)
                    {
                        if (dst[i].file.contains(labelSet[idx]))
                        {
                            QVariant currentProp = dst[i].file.value(labelSet[idx]);
                            QList<QVariant> currentPropList;

                            if (currentProp.canConvert<QList<QVariant> >() )
                            {
                                currentPropList = currentProp.toList();
                            }
                            else if (currentProp.canConvert<QRectF>())
                            {
                                currentPropList.append(currentProp);
                            }
                            else
                            {
                                qFatal("Unknown type of property");
                            }

                            currentPropList.append(rectSet[idx]);
                            dst[i].file.set(labelSet[idx], QVariant::fromValue(currentPropList));
                        }
                        else
                        {
                            if (labels.isEmpty()) dst[i].file.appendRect(rectSet[idx]);
                            else dst[i].file.set(labelSet[idx], rectSet[idx]);
                        }
                    }
                }

            }
        }
    }
    RectMarkingWindow *trueWindow;
    void init()
    {
        if (!Globals->useGui)
            return;
        initActual<RectMarkingWindow>();
        trueWindow = dynamic_cast<RectMarkingWindow *> (this->window);
        trueWindow->setKeys(this->keys);
    }
};

BR_REGISTER(Transform, ManualRectsTransform)

/*!
 * \ingroup transforms
 * \brief Elicits metadata for templates in a pretty GUI
 * \author Scott Klum \cite sklum
 */
class ElicitTransform : public ShowTransform
{
    Q_PROPERTY(QStringList keys READ get_keys WRITE set_keys RESET reset_keys STORED false)
    BR_PROPERTY(QStringList, keys, QStringList())

    Q_OBJECT

    GUIWindow *gui;

public:
    ElicitTransform() : ShowTransform()
    {
        gui = NULL;
    }

    ~ElicitTransform()
    {
        delete gui;
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        Transform *non_const = (ElicitTransform *) this;
        non_const->projectUpdate(src,dst);
    }

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;

        if (src.empty()) return;

        for (int i = 0; i < dst.size(); i++) {
            foreach (const cv::Mat &m, dst[i]) {
                qImageBuffer = toQImage(m);
                displayBuffer->convertFromImage(qImageBuffer);

                emit updateImage(displayBuffer->copy(displayBuffer->rect()));

                QStringList metadata = gui->waitForButtonPress();
                for (int j = 0; j < keys.size(); j++) dst[i].file.set(keys[j],metadata[j]);
            }
        }
    }

    void init()
    {
        initActual<GUIWindow>();
    }

    template<typename GUIType>
    void initActual()
    {
        if (!Globals->useGui) {
            qWarning("GUI transform %s created without enabling GUI support.\nRun \"br -gui ...\" to enable GUI support from the command line, or set\nGlobals->useGui to true.", this->metaObject()->className());
            return;
        }

        TimeVaryingTransform::init();

        if (displayBuffer)
            delete displayBuffer;

        displayBuffer = new QPixmap();

        if (gui)
            delete gui;

        gui = creator.getItem<GUIType>();
        gui->setKeys(keys);
        // Connect our signals to the window's slots
        connect(this, SIGNAL(updateImage(QPixmap)), gui,SLOT(showImage(QPixmap)));
        connect(this, SIGNAL(hideWindow()), gui, SLOT(hide()));
    }
};
BR_REGISTER(Transform, ElicitTransform)

/*!
 * \ingroup transforms
 * \brief Display an image, and asks a yes/no question about it
 * \author Charles Otto \cite caotto
 */
class SurveyTransform : public ShowTransform
{
    Q_OBJECT

public:
    Q_PROPERTY(QString question READ get_question WRITE set_question RESET reset_question STORED false)
    BR_PROPERTY(QString, question, "Yes/No")

    Q_PROPERTY(QString propertyName READ get_propertyName WRITE set_propertyName RESET reset_propertyName STORED false)
    BR_PROPERTY(QString, propertyName, "answer")


    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (Globals->parallelism > 1)
            qFatal("SurveyTransform cannot execute in parallel.");

        dst = src;

        if (src.empty())
            return;

        for (int i = 0; i < dst.size(); i++) {
            foreach (const cv::Mat &m, dst[i]) {
                qImageBuffer = toQImage(m);
                displayBuffer->convertFromImage(qImageBuffer);

                emit updateImage(displayBuffer->copy(displayBuffer->rect()));

                // Blocking wait for a key-press
                if (this->waitInput) {
                    QString answer = p_window->waitForKeyPress();

                    dst[i].file.set(this->propertyName, answer);
                }
            }
        }
    }
    PromptWindow *p_window;


    void init()
    {
        if (!Globals->useGui)
            return;

        initActual<PromptWindow>();
        p_window = (PromptWindow *) window;

        emit changeTitle(this->question);
    }
};

BR_REGISTER(Transform, SurveyTransform)

/*!
 * \brief DOCUMENT ME
 * \author Unknown \cite unknown
 */
class FilterTransform : public ShowTransform
{
    Q_OBJECT

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (Globals->parallelism > 1)
            qFatal("FilterTransform cannot execute in parallel.");

        if (src.empty())
            return;

        foreach (const Template &t, src) {
            Template u(t.file);
            foreach (const cv::Mat &m, t) {
                qImageBuffer = toQImage(m);
                displayBuffer->convertFromImage(qImageBuffer);

                emit updateImage(displayBuffer->copy(displayBuffer->rect()));

                // Blocking wait for a key-press
                if (this->waitInput) {
                    QString answer = p_window->waitForKeyPress();
                    qDebug() << answer;
                    if (answer == "y")
                        u.append(m);
                }
            }
            if (!u.empty())
                dst.append(u);
        }
    }
    PromptWindow *p_window;


    void init()
    {
        if (!Globals->useGui)
            return;

        initActual<PromptWindow>();
        p_window = (PromptWindow *) window;

        emit changeTitle("Keep: y Discard: n");
    }
};

BR_REGISTER(Transform, FilterTransform)

/*!
 * \ingroup transforms
 * \brief Limits the frequency of projects going through this transform to the input targetFPS
 * \author Charles Otto \cite caotto
 */
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

    void finalize(TemplateList &output)
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

/*!
 * \ingroup transforms
 * \brief Calculates the average FPS of projects going through this transform, stores the result in AvgFPS
 * Reports an average FPS from the initialization of this transform onwards.
 * \author Charles Otto \cite caotto
 */
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

    void finalize(TemplateList &output)
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

#include "gui/show.moc"
