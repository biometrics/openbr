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

#include <QElapsedTimer>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief DOCUMENT ME CHARLES
 * \author Unknown \cite unknown
 */
class ProgressCounterTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(qint64 totalProgress READ get_totalProgress WRITE set_totalProgress RESET reset_totalProgress STORED false)
    BR_PROPERTY(qint64, totalProgress, 1)

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;

        qint64 elapsed = timer.elapsed();
        int last_frame = -2;
        if (!dst.empty()) {
            for (int i=0;i < dst.size();i++) {
                int frame = dst[i].file.get<int>("FrameNumber", -1);
                if (frame == last_frame && frame != -1)
                    continue;

                // Use 1 as the starting index for progress output
                Globals->currentProgress = dst[i].file.get<qint64>("progress",0)+1;
                dst[i].file.remove("progress");
                last_frame = frame;

                Globals->currentStep++;
            }
        }

        // updated every second
        if (elapsed > 1000) {
            Globals->printStatus();
            timer.start();
        }

        return;
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }

    void finalize(TemplateList &data)
    {
        (void) data;
        float p = br_progress();
        qDebug("\r%05.2f%%  ELAPSED=%s  REMAINING=%s  COUNT=%g", p*100, QtUtils::toTime(Globals->startTime.elapsed()/1000.0f).toStdString().c_str(), QtUtils::toTime(0).toStdString().c_str(), Globals->currentStep);
        timer.start();
        Globals->startTime.start();
        Globals->currentStep = 0;
        Globals->currentProgress = 0;
        Globals->totalSteps = totalProgress;
    }

    void init()
    {
        timer.start();
        Globals->startTime.start();
        Globals->currentProgress = 0;
        Globals->currentStep = 0;
        Globals->totalSteps = totalProgress;
    }

public:
    ProgressCounterTransform() : TimeVaryingTransform(false,false) {}
    QElapsedTimer timer;
};

BR_REGISTER(Transform, ProgressCounterTransform)

} // namespace br

#include "core/progresscounter.moc"
