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

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Gives time elapsed over a specified Transform as a function of both images (or frames) and pixels.
 * \author Jordan Cheney \cite JordanCheney
 * \author Josh Klontz \cite jklontz
 */
class StopWatchTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString description READ get_description WRITE set_description RESET reset_description)
    BR_PROPERTY(QString, description, "Identity")

    br::Transform *transform;
    mutable QMutex mutex;
    mutable long miliseconds;
    mutable long images;
    mutable long pixels;

public:
    StopWatchTransform()
    {
        reset();
    }

private:
    void reset()
    {
        miliseconds = 0;
        images = 0;
        pixels = 0;
    }

    void init()
    {
        transform = Transform::make(description);
    }

    void train(const QList<TemplateList> &data)
    {
        transform->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        QTime watch;
        watch.start();
        transform->project(src, dst);

        QMutexLocker locker(&mutex);
        miliseconds += watch.elapsed();
        images++;
        foreach (const cv::Mat &m, src)
            pixels += (m.rows * m.cols);
    }

    void finalize(TemplateList &)
    {
        qDebug("\nProfile for \"%s\"\n"
               "\tSeconds: %g\n"
               "\tTemplates/s: %g\n"
               "\tPixels/s: %g\n",
               qPrintable(description),
               miliseconds / 1000.0,
               images * 1000.0 / miliseconds,
               pixels * 1000.0 / miliseconds);
        reset();
    }

    void store(QDataStream &stream) const
    {
        transform->store(stream);
    }

    void load(QDataStream &stream)
    {
        transform->load(stream);
    }
};

BR_REGISTER(Transform, StopWatchTransform)

} // namespace br

#include "metadata/stopwatch.moc"
