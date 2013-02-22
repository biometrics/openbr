#include <stasm_dll.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <openbr_plugin.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup initializers
 * \brief Initialize Stasm
 * \author Scott Klum \cite sklum
 */
class StasmInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        Globals->abbreviations.insert("RectFromStasmEyes","RectFromLandmarks([27, 28, 29, 30, 31, 32, 33, 34, 35, 36],10)");
        Globals->abbreviations.insert("RectFromStasmJaw","RectFromLandmarks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],10)");
        Globals->abbreviations.insert("RectFromStasmBrow","RectFromLandmarks([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],10)");
        Globals->abbreviations.insert("RectFromStasmNose","RectFromLandmarks([37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],10)");
        Globals->abbreviations.insert("RectFromStasmMouth","RectFromLandmarks([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],10)");
    }
};

BR_REGISTER(Initializer, StasmInitializer)

/*!
 * \ingroup transforms
 * \brief Wraps STASM key point detector
 * \author Scott Klum \cite sklum
 */

class StasmTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        int nlandmarks;
        int landmarks[500];

        AsmSearchDll(&nlandmarks, landmarks,
                     src.file.name.toStdString().c_str(), reinterpret_cast<char*>(src.m().data), src.m().cols, src.m().rows,
                     src.m(), (src.m().channels() == 3), qPrintable(Globals->sdkPath + "/share/openbr/models/stasm/mu-68-1d.conf"),  qPrintable(Globals->sdkPath + "/share/openbr/models/stasm/mu-76-2d.conf"));

        if (nlandmarks == 0) qFatal("Unable to detect Stasm landmarks");

        dst = src;
        for (int i = 0; i < nlandmarks; i++)
            dst.file.appendLandmark(QPointF(landmarks[2 * i], landmarks[2 * i + 1]));
    }
};

BR_REGISTER(Transform, StasmTransform)

} // namespace br

#include "stasm.moc"
