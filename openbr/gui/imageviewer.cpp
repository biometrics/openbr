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

#include "imageviewer.h"

/*** PUBLIC ***/
br::ImageViewer::ImageViewer(QWidget *parent)
    : QLabel(parent)
{
    setAlignment(Qt::AlignCenter);
    setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
}

void br::ImageViewer::setDefaultText(const QString &text)
{
    defaultText = text;
}

void br::ImageViewer::setImage(const QString &file, bool async)
{
    if(file.isNull()) src = QImage(); // Gets rid of runtime FileEngine::open warning
    else src = QImage(file);
    updatePixmap(async);
}

void br::ImageViewer::setImage(const QImage &image, bool async)
{
	src = image.copy();
    updatePixmap(async);
}

void br::ImageViewer::setImage(const QPixmap &pixmap, bool async)
{
    src = pixmap.toImage();
    updatePixmap(async);
}

/*** PRIVATE ***/
void br::ImageViewer::updatePixmap(bool async)
{
    if (async) {
        QTimer::singleShot(0, this, SLOT(updatePixmap()));
        return;
    }

	QMutexLocker locker(&mutex);
    if (src.isNull() || size().isNull()) {
        QLabel::setPixmap(QPixmap());
        QLabel::setText(defaultText);
    } else {
		QLabel::clear();
        QLabel::setPixmap(QPixmap::fromImage(src.scaled(size(), Qt::KeepAspectRatio)));
    }
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
    updatePixmap();
}
