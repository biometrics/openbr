#include <stasm_dll.hpp>
#include <stasm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "openbr_internal.h"

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
        Globals->abbreviations.insert("RectFromStasmEyes","RectFromPoints([27, 28, 29, 30, 31, 32, 33, 34, 35, 36],0.125,6.0)+Resize(44,164)");
        Globals->abbreviations.insert("RectFromStasmJaw","RectFromPoints([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],10)");
        Globals->abbreviations.insert("RectFromStasmBrow","RectFromPoints([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],0.15,6)+Resize(28,132)");
        Globals->abbreviations.insert("RectFromStasmNose","RectFromPoints([38, 39, 40, 41, 42, 43, 44, 67],0.15,1.25)+Resize(44,44)");
        Globals->abbreviations.insert("RectFromStasmMouth","RectFromPoints([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],0.3,3.0)+Resize(28,68)");
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

    QList<ASM_MODEL> models;
    //mutable QMutex mutex;

    void init()
    {
        models = nInitAsmModels(qPrintable(Globals->sdkPath + "/share/openbr/models/stasm/mu-68-1d.conf"),  qPrintable(Globals->sdkPath + "/share/openbr/models/stasm/mu-76-2d.conf"));
    }

    void project(const Template &src, Template &dst) const
    {
        //QMutexLocker locker(&mutex);

        //qDebug() << "PittPatt X: " << src.file.get<QPoint>("Affine_0").x() - src.m().cols/2 << src.file.get<QPoint>("Affine_1").x() - src.m().cols/2;
        //qDebug() << "PittPatt Y: " << src.m().rows/2 - src.file.get<QPoint>("Affine_0").y() << src.m().rows/2 - src.file.get<QPoint>("Affine_1").y();

        qDebug() << src.file.fileName();

        int numLandmarks;
        int landmarks[500];

        DET_PARAMS parameters;
        parameters.lex = src.file.get<QPoint>("Affine_0").x() - src.m().cols/2;
        parameters.ley = src.m().rows/2 - src.file.get<QPoint>("Affine_0").y();
        parameters.rex = src.file.get<QPoint>("Affine_1").x() - src.m().cols/2;
        parameters.rey = src.m().rows/2 - src.file.get<QPoint>("Affine_1").y();

        AsmSearchDll(numLandmarks, landmarks, qPrintable(src.file.name),
                     reinterpret_cast<char*>(src.m().data), src.m(), models,
                     qPrintable(Globals->sdkPath + "/share/openbr/models/stasm/"),
                     parameters);

        if (numLandmarks == 0) {
            qWarning("Unable to detect Stasm landmarks for %s", qPrintable(src.file.fileName()));
            dst.file.set("FTE", true);
            dst.m() = src.m();
            return;
        }

        for (int i = 0; i < numLandmarks; i++) {
            dst.file.appendPoint(QPointF(landmarks[2 * i], landmarks[2 * i + 1]));
        }

        dst.m() = src.m();
    }
};

BR_REGISTER(Transform, StasmTransform)

} // namespace br

#include "stasm.moc"
