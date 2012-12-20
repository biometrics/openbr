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

#ifndef __IMAGEVIEWER_H
#define __IMAGEVIEWER_H

#include <QImage>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QPixmap>
#include <QResizeEvent>
#include <QString>
#include <QWidget>

#include <openbr_export.h>

namespace br
{

class BR_EXPORT_GUI ImageViewer : public QLabel
{
    Q_OBJECT
    QString defaultText;
    QImage src;

public:
    explicit ImageViewer(QWidget *parent = 0);
    void setDefaultText(const QString &text, bool async = false);
    void setImage(const QString &file, bool async = false);
    void setImage(const QImage &image, bool async = false);
    void setImage(const QPixmap &pixmap, bool async = false);
    bool isNull() const { return src.isNull(); }
    int imageWidth() const { return src.width(); }
    int imageHeight() const { return src.height(); }

protected slots:
    void keyPressEvent(QKeyEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void resizeEvent(QResizeEvent *event);

private slots:
    void updatePixmap(bool async = false);
};

} // namespace br

#endif // __IMAGEVIEWER_H
