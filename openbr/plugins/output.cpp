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

#include <QByteArray>
#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QHash>
#include <QFile>
#include <QFileInfo>
#include <QFutureSynchronizer>
#include <QList>
#include <QMutex>
#include <QPair>
#include <QVector>
#include <QtConcurrent>
#include <QtGlobal>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <limits>
#include <assert.h>
#include "openbr_internal.h"

#include "openbr/core/bee.h"
#include "openbr/core/eval.h"
#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

namespace br
{
/*!
 * \ingroup outputs
 * \brief Adaptor class -- write a matrix output using Format classes.
 * \author Charles Otto \cite caotto
 */
class DefaultOutput : public MatrixOutput
{
    Q_OBJECT

    ~DefaultOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;

        br::Template T(this->file, this->data);
        QScopedPointer<Format> writer(Factory<Format>::make(this->file));
        writer->write(T);
    }
};

BR_REGISTER(Output, DefaultOutput)

/*!
 * \ingroup outputs
 * \brief Comma separated values output.
 * \author Josh Klontz \cite jklontz
 */
class csvOutput : public MatrixOutput
{
    Q_OBJECT

    ~csvOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        QStringList lines;
        if (Globals->crossValidate == 0) {
            for (int i=0; i<queryFiles.size(); i++) {
                QStringList words;
                for (int j=0; j<targetFiles.size(); j++)
                    words.append(queryFiles[i].name+","+targetFiles[j].baseName() + "," + toString(i,j)); // The toString idiom is used to output match scores - see MatrixOutput
                lines.append(words.join("\n"));
            }
            QtUtils::writeFile(file.name, lines);
        } else {
            for (int k=0; k<Globals->crossValidate; k++) {
                lines.clear();
                for (int i=0; i<queryFiles.size(); i++) {
                    int queryPartition = queryFiles[i].get<int>("Partition");
                    if (queryPartition != k) continue;
                    QStringList words;
                    QList<int> targetPartitions = targetFiles.crossValidationPartitions();
                    for (int j=0; j<targetFiles.size(); j++)
                        if (queryPartition == targetPartitions[j]) words.append(queryFiles[i].name+","+targetFiles[j].baseName() + "," + toString(i,j)); // The toString idiom is used to output match scores - see MatrixOutput
                        lines.append(words.join("\n"));
                }
                QtUtils::writeFile(file.name.arg(QString::number(k)), lines);
            }
        }
    }
};

BR_REGISTER(Output, csvOutput)

/*!
 * \ingroup outputs
 * \brief Matrix-like output for heat maps.
 * \author Scott Klum \cite sklum
 */
class heatOutput : public MatrixOutput
{
    Q_OBJECT
    Q_PROPERTY(int patches READ get_patches WRITE set_patches RESET reset_patches STORED false)
    BR_PROPERTY(int, patches, -1)

    ~heatOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;

        QStringList lines;
        for (int i=0; i<data.rows; i++) {
            lines.append(toString(i,0));
        }
        QtUtils::writeFile(file, lines);
    }

    void initialize(const FileList &targetFiles, const FileList &queryFiles)
    {
        if (patches == -1) qFatal("Heat output requires the number of patches");
        Output::initialize(targetFiles, queryFiles);
        data.create(patches, 1, CV_32FC1);
    }
};

BR_REGISTER(Output, heatOutput)

/*!
 * \ingroup outputs
 * \brief One score per row.
 * \author Josh Klontz \cite jklontz
 */
class meltOutput : public MatrixOutput
{
    Q_OBJECT

    ~meltOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        const bool genuineOnly = file.contains("Genuine") && !file.contains("Impostor");
        const bool impostorOnly = file.contains("Impostor") && !file.contains("Genuine");

        QMap<QString,QVariant> args = file.localMetadata();
        args.remove("Genuine");
        args.remove("Impostor");

        QString keys; foreach (const QString &key, args.keys()) keys += "," + key;
        QString values; foreach (const QVariant &value, args.values()) values += "," + value.toString();

        QStringList lines;
        if (file.baseName() != "terminal") lines.append(QString("Query,Target,Mask,Similarity%1").arg(keys));

        QList<QString> queryLabels = File::get<QString>(queryFiles, "Label");
        QList<QString> targetLabels = File::get<QString>(targetFiles, "Label");

        for (int i=0; i<queryFiles.size(); i++) {
            for (int j=(selfSimilar ? i+1 : 0); j<targetFiles.size(); j++) {
                const bool genuine = queryLabels[i] == targetLabels[j];
                if ((genuineOnly && !genuine) || (impostorOnly && genuine)) continue;
                lines.append(QString("%1,%2,%3,%4%5").arg(queryFiles[i],
                                                          targetFiles[j],
                                                          QString::number(genuine),
                                                          QString::number(data.at<float>(i,j)),
                                                          values));
            }
        }

        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, meltOutput)

/*!
 * \ingroup outputs
 * \brief \ref simmat output.
 * \author Josh Klontz \cite jklontz
 */
class mtxOutput : public Output
{
    Q_OBJECT
    int headerSize, rowBlock, columnBlock;
    cv::Mat blockScores;

    ~mtxOutput()
    {
        writeBlock();
    }

    void setBlock(int rowBlock, int columnBlock)
    {
        if ((rowBlock == 0) && (columnBlock == 0)) {
            // Initialize the file
            QFile f(file);
            QtUtils::touchDir(f);
            if (!f.open(QFile::WriteOnly))
                qFatal("Unable to open %s for writing.", qPrintable(file));
            const int endian = 0x12345678;
            QByteArray header;
            header.append("S2\n");
            header.append(qPrintable(targetFiles.first().get<QString>("Gallery", "Unknown_Target")));
            header.append("\n");
            header.append(qPrintable(queryFiles.first().get<QString>("Gallery", "Unknown_Query")));
            header.append("\nMF ");
            header.append(qPrintable(QString::number(queryFiles.size())));
            header.append(" ");
            header.append(qPrintable(QString::number(targetFiles.size())));
            header.append(" ");
            header.append(QByteArray((const char*)&endian, 4));
            header.append("\n");
            headerSize = f.write(header);
            const float defaultValue = -std::numeric_limits<float>::max();
            for (int i=0; i<targetFiles.size()*queryFiles.size(); i++)
                f.write((const char*)&defaultValue, 4);
            f.close();
        } else {
            writeBlock();
        }

        this->rowBlock = rowBlock;
        this->columnBlock = columnBlock;
        blockScores = cv::Mat(std::min(queryFiles.size()-rowBlock*Globals->blockSize, Globals->blockSize),
                              std::min(targetFiles.size()-columnBlock*Globals->blockSize, Globals->blockSize),
                              CV_32FC1);
    }

    void setRelative(float value, int i, int j)
    {
        blockScores.at<float>(i,j) = value;
    }

    void set(float value, int i, int j)
    {
        (void) value; (void) i; (void) j;
        qFatal("Logic error.");
    }

    void writeBlock()
    {
        QFile f(file);
        if (!f.open(QFile::ReadWrite))
            qFatal("Unable to open %s for modifying.", qPrintable(file));
        for (int i=0; i<blockScores.rows; i++) {
            f.seek(headerSize + sizeof(float)*(quint64(rowBlock*Globals->blockSize+i)*targetFiles.size()+(columnBlock*Globals->blockSize)));
            f.write((const char*)blockScores.row(i).data, sizeof(float)*blockScores.cols);
        }
        f.close();
    }
};

BR_REGISTER(Output, mtxOutput)

/*!
 * \ingroup outputs
 * \brief Rank retrieval output.
 * \author Josh Klontz \cite jklontz
 * \author Scott Klum \cite sklum
 */
class rrOutput : public MatrixOutput
{
    Q_OBJECT

    ~rrOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        const int limit = file.get<int>("limit", 20);
        const bool byLine = file.getBool("byLine");
        const bool simple = file.getBool("simple");
        const float threshold = file.get<float>("threshold", -std::numeric_limits<float>::max());

        QStringList lines;

        for (int i=0; i<queryFiles.size(); i++) {
            QStringList files;
            if (!byLine) files.append(queryFiles[i]);

            typedef QPair<float,int> Pair;
            foreach (const Pair &pair, Common::Sort(OpenCVUtils::matrixToVector<float>(data.row(i)), true, limit)) {
                if (pair.first < threshold) break;
                File target = targetFiles[pair.second];
                target.set("Score", QString::number(pair.first));
                if (simple) files.append(target.baseName() + " " + QString::number(pair.first));
                else files.append(target.flat());
            }
            lines.append(files.join(byLine ? "\n" : ","));
        }

        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, rrOutput)

/*!
 * \ingroup outputs
 * \brief Text file output.
 * \author Josh Klontz \cite jklontz
 */
class txtOutput : public MatrixOutput
{
    Q_OBJECT

    ~txtOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        QStringList lines;
        foreach (const File &file, queryFiles)
            lines.append(file.name + " " + file.get<QString>("Label"));
        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, txtOutput)

/*!
 * \ingroup outputs
 * \brief Output to the terminal.
 * \author Josh Klontz \cite jklontz
 */
class EmptyOutput : public MatrixOutput
{
    Q_OBJECT

    static QString bufferString(const QString &string, int length)
    {
        if (string.size() >= length)
            return string.left(length);
        QString buffer; buffer.fill(' ', length-string.size());
        return string+buffer;
    }

    ~EmptyOutput()
    {
        if (targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        QString result;
        if ((queryFiles.size() == 1) && (targetFiles.size() == 1)) {
            result = toString(0,0) + "\n";
        } else {
            const int CELL_SIZE = 12;

            result = bufferString(" ", CELL_SIZE) + " ";
            foreach (const QString &targetName, targetFiles.names())
                result += bufferString(targetName, CELL_SIZE) + " ";
            result += "\n";

            for (int i=0; i<queryFiles.size(); i++) {
                result += bufferString(queryFiles[i].name, CELL_SIZE) + " ";
                for (int j=0; j<targetFiles.size(); j++)
                    result += bufferString(toString(i,j), CELL_SIZE) + " ";
                result += "\n";
            }
        }

        printf("%s", qPrintable(result));
    }
};

BR_REGISTER(Output, EmptyOutput)

/*!
 * \ingroup outputs
 * \brief Evaluate the output matrix.
 * \author Josh Klontz \cite jklontz
 */
class evalOutput : public MatrixOutput
{
    Q_OBJECT
    Q_PROPERTY(bool crossValidate READ get_crossValidate WRITE set_crossValidate RESET reset_crossValidate STORED false)
    BR_PROPERTY(bool, crossValidate, true)

    ~evalOutput()
    {
        if (data.data) {
            const QString csv = QString(file.name).replace(".eval", ".csv");
            if ((Globals->crossValidate == 0) || (!crossValidate)) {
                Evaluate(data, BEE::makeMask(targetFiles, queryFiles), csv);
            } else {
                QFutureSynchronizer<float> futures;
                for (int i=0; i<Globals->crossValidate; i++)
                    futures.addFuture(QtConcurrent::run(Evaluate, data, targetFiles, queryFiles, csv.arg(QString::number(i)), i));
                futures.waitForFinished();

                QList<float> TARs;
                foreach (const QFuture<float> &future, futures.futures())
                    TARs.append(future.result());

                double mean, stddev;
                Common::MeanStdDev(TARs, &mean, &stddev);
                qDebug("TAR @ FAR = 0.001: %.3f +/- %.3f", mean, stddev);
            }
        }
    }
};

BR_REGISTER(Output, evalOutput)

/*!
 * \ingroup outputs
 * \brief Discards the scores.
 * \author Josh Klontz \cite jklontz
 */
class nullOutput : public Output
{
    Q_OBJECT

    void set(float value, int i, int j)
    {
        (void) value; (void) i; (void) j;
    }
};

BR_REGISTER(Output, nullOutput)

/*!
 * \ingroup outputs
 * \brief Outputs highest ranked matches with scores.
 * \author Scott Klum \cite sklum
 */
class rankOutput : public MatrixOutput
{
    Q_OBJECT

    ~rankOutput()
    {
        if (targetFiles.isEmpty() || queryFiles.isEmpty()) return;

        QList<int> ranks;
        QList<int> positions;
        QList<float> scores;
        QStringList lines;

        for (int i=0; i<queryFiles.size(); i++) {
            typedef QPair<float,int> Pair;
            int rank = 1;
            foreach (const Pair &pair, Common::Sort(OpenCVUtils::matrixToVector<float>(data.row(i)), true)) {
                if (Globals->crossValidate > 0 ? (targetFiles[pair.second].get<int>("Partition",-1) == queryFiles[i].get<int>("Partition",-1)) : true) {
                    if (QString(targetFiles[pair.second]) != QString(queryFiles[i])) {
                        if (targetFiles[pair.second].get<QString>("Label") == queryFiles[i].get<QString>("Label")) {
                            ranks.append(rank);
                            positions.append(pair.second);
                            scores.append(pair.first);
                            break;
                        }
                        rank++;
                    }
                }
            }
        }

        typedef QPair<int,int> RankPair;
        foreach (const RankPair &pair, Common::Sort(ranks, false))
            // pair.first == rank retrieved, pair.second == original position
            lines.append(queryFiles[pair.second].name + " " + QString::number(pair.first) + " " + QString::number(scores[pair.second]) + " " + targetFiles[positions[pair.second]].name);


        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, rankOutput)

/*!
 * \ingroup outputs
 * \brief The highest scoring matches.
 * \author Josh Klontz \cite jklontz
 */
class tailOutput : public Output
{
    Q_OBJECT

    struct Comparison
    {
        br::File query, target;
        float value;

        Comparison(const br::File &_query, const br::File &_target, float _value)
            : query(_query), target(_target), value(_value) {}

        QString toString(bool args) const
        {
            return QString::number(value) + "," + (args ? target.flat() : (QString)target) + "," + (args ? query.flat() : (QString)query);
        }

        bool operator<(const Comparison &other) const
        {
            return value < other.value;
        }
    };

    float threshold;
    int atLeast, atMost;
    bool args;
    float lastValue;
    QList<Comparison> comparisons;
    QMutex comparisonsLock;

    ~tailOutput()
    {
        if (file.isNull() || comparisons.isEmpty()) return;
        QStringList lines; lines.reserve(comparisons.size()+1);
        lines.append("Value,Target,Query");
        foreach (const Comparison &duplicate, comparisons)
            lines.append(duplicate.toString(args));
        QtUtils::writeFile(file, lines);
    }

    void initialize(const FileList &targetFiles, const FileList &queryFiles)
    {
        Output::initialize(targetFiles, queryFiles);
        threshold = file.get<float>("threshold", -std::numeric_limits<float>::max());
        atLeast = file.get<int>("atLeast", 1);
        atMost = file.get<int>("atMost", std::numeric_limits<int>::max());
        args = file.get<bool>("args", false);
        lastValue = -std::numeric_limits<float>::max();
    }

    void set(float value, int i, int j)
    {
        // Return early for self similar matrices
        if (selfSimilar && (i <= j)) return;

        // Consider only values passing the criteria
        if ((value < threshold) && (value <= lastValue) && (comparisons.size() >= atLeast))
            return;

        comparisonsLock.lock();
        if (comparisons.isEmpty() || (value < comparisons.last().value)) {
            // Special cases
            comparisons.append(Comparison(queryFiles[i], targetFiles[j], value));
        } else {
            // General case
            for (int k=0; k<comparisons.size(); k++) {
                if (comparisons[k].value < value) {
                    comparisons.insert(k, Comparison(queryFiles[i], targetFiles[j], value));
                    break;
                }
            }
        }

        while (comparisons.size() > atMost)
            comparisons.removeLast();
        while ((comparisons.size() > atLeast) && (comparisons.last().value < threshold))
            comparisons.removeLast();
        lastValue = comparisons.last().value;
        comparisonsLock.unlock();
    }
};

BR_REGISTER(Output, tailOutput)

/*!
 * \ingroup outputs
 * \brief The highest scoring matches.
 * \author Josh Klontz \cite jklontz
 */
class bestOutput : public Output
{
    Q_OBJECT

    typedef QPair< float, QPair<int, int> > BestMatch;
    QList<BestMatch> bestMatches;

    ~bestOutput()
    {
        if (file.isNull() || bestMatches.isEmpty()) return;
        qSort(bestMatches);
        QStringList lines; lines.reserve(bestMatches.size()+1);
        lines.append("Value,Target,Query");
        for (int i=bestMatches.size()-1; i>=0; i--)
            lines.append(QString::number(bestMatches[i].first) + "," + targetFiles[bestMatches[i].second.second] + "," + queryFiles[bestMatches[i].second.first]);
        QtUtils::writeFile(file, lines);
    }

    void initialize(const FileList &targetFiles, const FileList &queryFiles)
    {
        Output::initialize(targetFiles, queryFiles);
        bestMatches.reserve(queryFiles.size());
        for (int i=0; i<queryFiles.size(); i++)
            bestMatches.append(BestMatch(-std::numeric_limits<float>::max(), QPair<int,int>(-1, -1)));
    }

    void set(float value, int i, int j)
    {
        static QMutex lock;

        // Return early for self similar matrices
        if (selfSimilar && (i == j)) return;

        if (value > bestMatches[i].first) {
            lock.lock();
            if (value > bestMatches[i].first)
                bestMatches[i] = BestMatch(value, QPair<int,int>(i,j));
            lock.unlock();
        }
    }
};

BR_REGISTER(Output, bestOutput)

/*!
 * \ingroup outputs
 * \brief Score histogram.
 * \author Josh Klontz \cite jklontz
 */
class histOutput : public Output
{
    Q_OBJECT

    float min, max, step;
    QVector<int> bins;

    ~histOutput()
    {
        if (file.isNull() || bins.isEmpty()) return;
        QStringList counts;
        foreach (int count, bins)
            counts.append(QString::number(count));
        const QString result = counts.join(",");
        QtUtils::writeFile(file, result);
    }

    void initialize(const FileList &targetFiles, const FileList &queryFiles)
    {
        Output::initialize(targetFiles, queryFiles);
        min = file.get<float>("min", -5);
        max = file.get<float>("max", 5);
        step = file.get<float>("step", 0.1);
        bins = QVector<int>((max-min)/step, 0);
    }

    void set(float value, int i, int j)
    {
        (void) i;
        (void) j;
        if ((value < min) || (value >= max)) return;
        bins[(value-min)/step]++; // This should technically be locked to ensure atomic increment
    }
};

BR_REGISTER(Output, histOutput)

} // namespace br

#include "output.moc"
