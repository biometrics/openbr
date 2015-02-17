#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Concatenates all input matrices by column into a single matrix.
 * Use after a fork to concatenate two feature matrices by column.
 * \author Austin Blanton \cite imaus10
 */
class CatColsTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        int half = src.size()/2;
        for (int i=0; i<half; i++) {
            Mat first = src[i];
            Mat second = src[half+i];
            Mat both;
            hconcat(first, second, both);
            dst.append(both);
        }
        dst.file = src.file;
    }
};

BR_REGISTER(Transform, CatColsTransform)

} // namespace br

#include "imgproc/catcols.moc"
