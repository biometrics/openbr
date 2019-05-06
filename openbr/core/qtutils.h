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

#ifndef QTUTILS_QTUTILS_H
#define QTUTILS_QTUTILS_H

#include <QBuffer>
#include <QByteArray>
#include <QDataStream>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFuture>
#include <QFutureSynchronizer>
#include <QMap>
#include <QString>
#include <QStringList>
#include <QThreadPool>
#include <QJsonObject>
#include <string>
#include <vector>

#include <openbr/openbr_export.h>

namespace QtUtils
{
    /**** File Utilities ****/
    QStringList readLines(const QString &file);
    void readFile(const QString &file, QStringList &lines);
    void readFile(const QString &file, QByteArray &data, bool uncompress = false);
    void writeFile(const QString &file, const QStringList &lines);
    void writeFile(const QString &file, const QString &data);
    void writeFile(const QString &file, const QByteArray &data, int compression = 0);
    void copyFile(const QString &src, const QString &dst);

    /**** Directory Utilities ****/
    void touchDir(const QDir &dir);
    void touchDir(const QFile &file);
    void touchDir(const QFileInfo &fileInfo);
    void emptyDir(QDir &dir);
    void deleteDir(QDir &dir);
    QString find(const QString &file, const QString &alt);
    QString getAbsolutePath(const QString &filename);

    /**** String Utilities ****/
    bool toBool(const QString &string, bool *ok = NULL);
    int toInt(const QString &string);
    float toFloat(const QString &string);
    QList<float> toFloats(const QStringList &strings);
    QStringList toStringList(const QList<float> &values);
    QStringList toStringList(const std::vector<std::string> &string_list);
    QStringList toStringList(int num_strings, const char* strings[]);
    QString shortTextHash(QString string);
    QStringList parse(QString args, char split = ',', bool *ok = NULL);
    void checkArgsSize(const QString &name, const QStringList &args, int min, int max);
    BR_EXPORT_ALWAYS QPointF toPoint(const QString &string, bool *ok = NULL);
    BR_EXPORT_ALWAYS QRectF toRect(const QString &string, bool *ok = NULL);
    QStringList naturalSort(const QStringList &strings);
    QString toTime(int s);

    /**** Process Utilities ****/
    bool runRScript(const QString &file);
    bool runDot(const QString &file);
    void showFile(const QString &file);

    /**** Variant Utilities ****/
    BR_EXPORT QString toString(const QVariant &variant);
    QString toString(const QVariantList &variantList);
    BR_EXPORT QString toString(const QVariantMap &QVariantMap);
    BR_EXPORT QStringList toStringList(const QVariantMap &QVariantMap);
    BR_EXPORT QVariant fromString(const QString &value);
    BR_EXPORT QJsonValue fromVariant(const QVariant &variant);
    BR_EXPORT QJsonObject fromVariantMap(const QVariantMap &variantMap);
    BR_EXPORT QVariant fromJsonValue(const QJsonValue &value);
    BR_EXPORT QVariantMap fromJsonObject(const QJsonObject &object);

    template <typename T>
    QVariantList toVariantList(const QList<T> &list)
    {
        QVariantList variantList;
        foreach (const T &item, list)
            variantList << item;

        return variantList;
    }

    /**** Point Utilities ****/
    float euclideanLength(const QPointF &point);
    float orientation(const QPointF &pointA, const QPointF &pointB);

    /**** Rect Utilities ****/
    float area(const QRectF &r);
    float overlap(const QRectF &r, const QRectF &s);
    
    class BlockCompression : public QIODevice
    {
    public:
        BlockCompression(QIODevice *_basis);
        BlockCompression();
        int blockSize;
        QIODevice *basis;

        bool open(QIODevice::OpenMode mode);

        void close();

        void setBasis(QIODevice *_basis);

        QDataStream blockReader;
        QByteArray decompressedBlock;
        QBuffer decompressedBlockReader;

        // read from current decompressed block, if out of space, read and decompress another
        // block from basis
        qint64 readData(char *data, qint64 remaining);

        bool isSequential() const;

        // write to a QByteArray, when max block sized is reached, compress and write
        // it to basis
        QBuffer precompressedBlockWriter;
        QDataStream blockWriter;
        qint64 writeData(const char *data, qint64 remaining);
    };
}

#endif // QTUTILS_QTUTILS_H
