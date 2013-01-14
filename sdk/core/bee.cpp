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

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QMap>
#include <QRegExp>
#ifndef BR_EMBEDDED
#include <QtXml>
#endif // BR_EMBEDDED
#include <algorithm>
#include <limits>
#include <openbr_plugin.h>

#include "bee.h"
#include "opencvutils.h"
#include "qtutils.h"

using namespace cv;
using namespace br;

/**** BEE ****/
FileList BEE::readSigset(QString sigset, bool ignoreMetadata)
{
    FileList fileList;

#ifndef BR_EMBEDDED
    QDomDocument doc(sigset);
    QFile file(sigset);
    bool success;
    success = file.open(QIODevice::ReadOnly); if (!success) qFatal("BEE::readSigset unable to open %s for reading.", qPrintable(sigset));
    success = doc.setContent(&file);          if (!success) qFatal("BEE::readSigset unable to parse %s.", qPrintable(sigset));
    file.close();

    QDomElement docElem = doc.documentElement();
    if (docElem.nodeName() != "biometric-signature-set")
        return fileList;

    QDomNode subject = docElem.firstChild();
    while (!subject.isNull()) {
        // Looping through subjects
        QDomNode fileNode = subject.firstChild();
        QDomElement d = subject.toElement();
        QString name = d.attribute("name");
        while (!fileNode.isNull()) {
            // Looping through files
            File file;

            QDomElement e = fileNode.toElement();
            QDomNamedNodeMap attributes = e.attributes();
            for (unsigned int i=0; i<attributes.length(); i++) {
                QString key = attributes.item(i).nodeName();
                QString value = attributes.item(i).nodeValue();

                if (key == "file-name") {
                    File newFile(value, name);
                    newFile.append(file);
                    file = newFile;
                } else if (!ignoreMetadata) {
                    file.insert(key, value);
                }
            }

            if (file.isNull()) qFatal("BEE::readSigset empty file-name in %s.", qPrintable(sigset));
            fileList.append(file);

            fileNode = fileNode.nextSibling();
        }
        subject = subject.nextSibling();
    }
#else // BR_EMBEDDED
    (void) sigset;
    (void) ignoreMetadata;
#endif // BR_EMBEDDED

    return fileList;
}

void BEE::writeSigset(const QString &sigset, const br::FileList &fileList)
{
    typedef QPair<QString,QString> Signature; // QPair<Subject, File>
    QList<Signature> signatures;

    foreach (const File &file, fileList)
        signatures.append(Signature(file.subject(), file.name));

    QFile file(sigset);
    file.open(QFile::WriteOnly);
    file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
               "<biometric-signature-set>\n");
    foreach (const Signature &signature, signatures)
        file.write(qPrintable(QString("\t<biometric-signature name=\"%1\">\n"
                                      "\t\t<presentation file-name=\"%2\"/>\n"
                                      "\t</biometric-signature>\n").arg(signature.first, signature.second)));
    file.write("</biometric-signature-set>\n");
    file.close();
}

template <typename T>
Mat readMatrix(const br::File &matrix)
{
    // Special case matrix construction
    if (matrix == "Matrix") {
        const int size = matrix.getInt("Size");
        const int step = matrix.getInt("Step", 1);
        if (size % step != 0) qFatal("bee.cpp readMatrix step does not divide size evenly.");

        if (sizeof(T) == sizeof(BEE::Mask_t)) {
            const bool selfSimilar = matrix.getBool("SelfSimilar");

            Mat m(size, size, CV_8UC1);
            m.setTo(BEE::NonMatch);
            for (int i=0; i<size; i+=step)
                for (int j=0; j<step; j++)
                    for (int k=0; k<step; k++)
                        m.at<BEE::Mask_t>(i+j,i+k) = ((selfSimilar && (j == k)) ? BEE::DontCare : BEE::Match);
            return m;
        } else if (sizeof(T) == sizeof(BEE::Simmat_t)) {
            Mat m(size, size, CV_32FC1);
            m.setTo(BEE::NonMatch);
            for (int i=0; i<size; i+=step)
                for (int j=0; j<step; j++)
                    for (int k=0; k<step; k++)
                        m.at<BEE::Simmat_t>(i+j,i+k) = 1;
            return m;
        }
    }

    QFile file(matrix);
    bool success = file.open(QFile::ReadOnly);
    if (!success) qFatal("bee.cpp readMatrix unable to open %s for reading.", qPrintable((QString)matrix));

    // Check format
    QByteArray format = file.readLine();
    bool isDistance = (format[0] == 'D');
    if (format[1] != '2') qFatal("bee.cpp readMatrix invalid matrix header.");

    // Skip sigset lines
    file.readLine();
    file.readLine();

    // Get matrix size
    QStringList words = QString(file.readLine()).split(" ");
    int rows = words[1].toInt();
    int cols = words[2].toInt();

    // Get matrix data
    qint64 bytesExpected = (qint64)rows*(qint64)cols*(qint64)sizeof(T);
    Mat m(rows, cols, OpenCVType<T,1>::make());
    if (file.read((char*)m.data, bytesExpected) != bytesExpected)
        qFatal("bee.cpp readMatrix invalid matrix size.");
    file.close();

    Mat result;
    if (isDistance ^ matrix.getBool("Negate")) m.convertTo(result, -1, -1);
    else                                       result = m.clone();
    return result;
}

Mat BEE::readSimmat(const br::File &simmat)
{
    return readMatrix<Simmat_t>(simmat);
}

Mat BEE::readMask(const br::File &mask)
{
    return readMatrix<Mask_t>(mask);
}

template <typename T>
void writeMatrix(const Mat &m, const QString &matrix, const QString &targetSigset, const QString &querySigset)
{
    if (m.type() != OpenCVType<T,1>::make()) qFatal("bee.cpp writeMatrix invalid matrix type.");

    int elemSize = sizeof(T);
    QString matrixType;
    if      (elemSize == 1) matrixType = "B";
    else if (elemSize == 4) matrixType = "F";
    else                    qFatal("bee.cpp writeMatrix invalid element size.\n");

    char buff[4];
    QFile file(matrix);
    bool success = file.open(QFile::WriteOnly); if (!success) qFatal("bee.cpp writeMatrix unable to open %s for writing.", qPrintable(matrix));
    file.write("S2\n");
    file.write(qPrintable(QFileInfo(targetSigset).fileName()));
    file.write("\n");
    file.write(qPrintable(QFileInfo(querySigset).fileName()));
    file.write("\n");
    file.write("M");
    file.write(qPrintable(matrixType));
    file.write(" ");
    file.write(qPrintable(QString::number(m.rows)));
    file.write(" ");
    file.write(qPrintable(QString::number(m.cols)));
    file.write(" ");
    int endian = 0x12345678;
    memcpy(&buff, &endian, 4);
    file.write(buff, 4);
    file.write("\n");
    file.write((const char*)m.data, m.rows*m.cols*elemSize);
    file.close();
}

void BEE::writeSimmat(const Mat &m, const QString &simmat, const QString &targetSigset, const QString &querySigset)
{
    writeMatrix<Simmat_t>(m, simmat, targetSigset, querySigset);
}

void BEE::writeMask(const Mat &m, const QString &mask, const QString &targetSigset, const QString &querySigset)
{
    writeMatrix<Mask_t>(m, mask, targetSigset, querySigset);
}

void BEE::makeMask(const QString &targetInput, const QString &queryInput, const QString &mask)
{
    qDebug("Making mask from %s and %s to %s", qPrintable(targetInput), qPrintable(queryInput), qPrintable(mask));

    TemplateList target(TemplateList::fromInput(targetInput));
    TemplateList query(TemplateList::fromInput(queryInput));
    FileList targetFiles = target.files();
    FileList queryFiles = query.files();
    QList<float> targetLabels = targetFiles.labels();
    QList<float> queryLabels = queryFiles.labels();

    Mat vals(queryFiles.size(), targetFiles.size(), CV_8UC1);
    for (int i=0; i<queryFiles.size(); i++) {
        const int labelA = queryLabels[i];
        const QString &fileA = queryFiles[i];
        for (int j=0; j<targetFiles.size(); j++) {
            const int labelB = targetLabels[j];
            const QString &fileB = targetFiles[j];
            Mask_t val;
            if      (fileA == fileB)   val = DontCare;
            else if (labelA == -1)     val = DontCare;
            else if (labelB == -1)     val = DontCare;
            else if (labelA == labelB) val = Match;
            else                       val = NonMatch;
            vals.at<Mask_t>(i,j) = val;
        }
    }
    writeMask(vals, mask, targetInput, queryInput);
}

void BEE::combineMasks(const QStringList &inputMasks, const QString &outputMask, const QString &method)
{
    qDebug("Combining %d masks to %s with method %s", inputMasks.size(), qPrintable(outputMask), qPrintable(method));

    bool AND = true;
    if      (method == "And") AND = true;
    else if (method == "Or")  AND = false;
    else                      qFatal("combineMasks invalid method");

    QList<Mat> masks;
    foreach (const QString &inputMask, inputMasks)
        masks.append(readMask(inputMask));
    if (masks.size() < 2) qFatal("BEE::mergeMasks expects at least two masks.");

    const int rows = masks.first().rows;
    const int columns = masks.first().cols;

    Mat combinedMask(rows, columns, CV_8UC1);
    for (int i=0; i<rows; i++) {
        for (int j=0; j<columns; j++) {
            int genuineCount = 0;
            int imposterCount = 0;
            int dontcareCount = 0;
            for (int k=0; k<masks.size(); k++) {
                switch (masks[k].at<Mask_t>(i,j)) {
                  case Match:
                    genuineCount++;
                    break;
                  case NonMatch:
                    imposterCount++;
                    break;
                  case DontCare:
                    dontcareCount++;
                    break;
                }
            }
            if ((genuineCount != 0) && (imposterCount != 0)) qFatal("BEE::combinedMasks comparison is both a genuine and an imposter.");

            Mask_t val;
            if      (genuineCount > 0)  val = Match;
            else if (imposterCount > 0) val = NonMatch;
            else                        val = DontCare;
            if (AND && (dontcareCount > 0)) val = DontCare;
            combinedMask.at<Mask_t>(i,j) = val;
        }
    }

    writeMask(combinedMask, outputMask, "Combined_Targets", "Combined_Queries");
}
