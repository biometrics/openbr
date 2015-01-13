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

#include <QFileDialog>
#include <QMutexLocker>
#include <QSizePolicy>
#include <QTimer>
#include <QDebug>

#include "imageviewer.h"

/*** PUBLIC ***/
br::ImageViewer::ImageViewer(QWidget *parent)
    : QLabel(parent),
    mutex(QMutex::Recursive)
{
    setAlignment(Qt::AlignCenter);
    setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
}

void br::ImageViewer::setDefaultText(const QString &text)
{
    defaultText = text;
    updatePixmap(QImage());
}

void br::ImageViewer::setImage(const QString &file, bool async)
{
    QMutexLocker locker(&mutex);
    src = file.isNull() ? QImage() : QImage(file);
    updatePixmap(src, async);
}

void br::ImageViewer::setImage(const QImage &image, bool async)
{
    QMutexLocker locker(&mutex);
	src = image.copy();
    updatePixmap(src, async);
}

void br::ImageViewer::setImage(const QPixmap &pixmap, bool async)
{
    QMutexLocker locker(&mutex);
    src = pixmap.toImage();
    updatePixmap(src, async);
}

/*** PRIVATE ***/
void br::ImageViewer::updatePixmap(QImage image, bool async)
{
    if (async) {
        QMetaObject::invokeMethod(this, "updatePixmap", Qt::QueuedConnection, Q_ARG(QImage, image), Q_ARG(bool, false));
        return;
    }

    QMutexLocker locker(&mutex);
    if (image.isNull() || size().isNull()) {
        setPixmap(QPixmap());
        setText(defaultText);
        setFrameShape(QLabel::StyledPanel);
    } else {
        clear();
        setPixmap(QPixmap::fromImage(image.scaled(size(), Qt::KeepAspectRatio)));
        setFrameShape(QLabel::NoFrame);
    }
}

QSize br::ImageViewer::sizeHint() const
{
    return src.isNull() ? QSize() : QSize(width(), (src.height() * width() + /* round up */ src.width() - 1) / src.width());
}

/*** PROTECTED SLOTS ***/
void br::ImageViewer::keyPressEvent(QKeyEvent *event)
{
    QLabel::keyPressEvent(event);

    if ((event->key() == Qt::Key_S) && (event->modifiers() == Qt::ControlModifier) && !src.isNull()) {
        event->accept();
        const QString fileName = QFileDialog::getSaveFileName(this, "Save Image");
        if (!fileName.isEmpty()) src.save(fileName);
    }
}

void br::ImageViewer::mouseMoveEvent(QMouseEvent *event)
{
    QLabel::mouseMoveEvent(event);
    event->accept();
    setFocus();
}

void br::ImageViewer::resizeEvent(QResizeEvent *event)
{
    QLabel::resizeEvent(event);
    event->accept();
    updatePixmap(src);
}
