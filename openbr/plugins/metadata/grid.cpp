#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Add landmarks to the template in a grid layout
 * \author Josh Klontz \cite jklontz
 */
class GridTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
    Q_PROPERTY(int columns READ get_columns WRITE set_columns RESET reset_columns STORED false)
    BR_PROPERTY(int, rows, 1)
    BR_PROPERTY(int, columns, 1)

    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> landmarks;
        const float row_step = 1.f * src.m().rows / rows;
        const float column_step = 1.f * src.m().cols / columns;
        for (float y=row_step/2; y<src.m().rows; y+=row_step)
            for (float x=column_step/2; x<src.m().cols; x+=column_step)
                landmarks.append(QPointF(x,y));
        dst = src;
        dst.file.setPoints(landmarks);
    }
};

BR_REGISTER(Transform, GridTransform)

} // namespace br

#include "metadata/grid.moc"
