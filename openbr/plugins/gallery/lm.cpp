#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

#include <likely.h>
#include <likely/opencv.hpp>

namespace br
{

/*!
 * \ingroup formats
 * \brief Likely matrix format
 *
 * www.liblikely.org
 * \author Josh Klontz \cite jklontz
 */
class lmGallery : public Gallery
{
    Q_OBJECT
    QList<cv::Mat> mats;

    ~lmGallery()
    {
        if (mats.empty())
            return;
        if (!mats.first().data)
            qFatal("Null first matrix");

        const int depth = mats.first().depth();
        const int channels = mats.first().channels();
        const int columns = mats.first().cols;
        const int rows = mats.first().rows;
        const int frames = mats.size();
        likely_type type = likelyFromOpenCVDepth(depth);
        if (channels > 1) type |= likely_multi_channel;
        if (columns  > 1) type |= likely_multi_column;
        if (rows     > 1) type |= likely_multi_row;
        if (frames   > 1) type |= likely_multi_frame;

        const likely_mat m = likely_new(type, channels, columns, rows, frames, NULL);
        const size_t step = (type & likely_depth) * channels * columns * rows / 8;
        for (size_t i=0; i<size_t(frames); i++) {
            const cv::Mat &mat = mats[i];
            if (!mat.isContinuous())
                qFatal("Expected continuous matrix data");
            if (mat.depth() != depth)
                qFatal("Depth mismatch");
            if (mat.channels() != channels)
                qFatal("Channel mismatch");
            if (mat.cols != columns)
                qFatal("Columns mismatch");
            if (mat.rows != rows)
                qFatal("Rows mismatch");
            memcpy(m->data + i * step, mat.data, step);
        }

        if (!likely_write(m, qPrintable(file.name)))
            qFatal("Write failed");
        likely_release_mat(m);
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        TemplateList templates;
        const likely_const_mat m = likely_read(qPrintable(file.name), likely_file_matrix, likely_void);
        const size_t step = (m->type & likely_depth) * m->channels * m->columns * m->rows / 8;
        for (size_t i=0; i<m->frames; i++)
            templates.append(cv::Mat(m->rows, m->columns, CV_MAKETYPE(likelyToOpenCVDepth(m->type), m->channels), (void*)(m->data + i * step)).clone());
        return templates;
    }

    void write(const Template &t)
    {
        mats.append(t);
    }
};

BR_REGISTER(Gallery, lmGallery)

} // namespace br

#include "gallery/lm.moc"
