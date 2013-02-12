#ifndef __LIKELY_H
#define __LIKELY_H

#include <stddef.h>
#include <stdint.h>
#include <openbr.h>

/*!
 * \defgroup likely Literate Kernel Library
 * \brief Experimental low-level API for programming image processing kernels on heterogeneous hardware architectures.
 */

/*!
 * \addtogroup likely
 *  @{
 */

/*!
 * \brief Likely Matrix
 * \author Josh Klontz \cite jklontz
 */
struct Matrix
{
    uint8_t *data; /*!< Data */
    uint32_t channels; /*!< Channels */
    uint32_t columns; /*!< Columns */
    uint32_t rows; /*!< Rows */
    uint32_t frames; /*!< Frames */
    uint16_t hash; /*!< Bits : 8
                       Floating : 1
                       Signed : 1
                       (Reserved) : 2
                       Single-channel : 1
                       Single-column : 1
                       Single-row : 1
                       Single-frame : 1 */

    enum Hash { Bits = 0x00FF,
                Floating = 0x0100,
                Signed = 0x0200,
                SingleChannel = 0x1000,
                SingleColumn = 0x2000,
                SingleRow = 0x4000,
                SingleFrame = 0x8000,
                u1  = 1,
                u8  = 8,
                u16 = 16,
                u32 = 32,
                u64 = 64,
                s8  = 8  + Signed,
                s16 = 16 + Signed,
                s32 = 32 + Signed,
                s64 = 64 + Signed,
                f16 = 16 + Floating + Signed,
                f32 = 32 + Floating + Signed,
                f64 = 64 + Floating + Signed };

    /*Matrix() : data(NULL), channels(0), columns(0), rows(0), frames(0), hash(0) {}

    Matrix(uint32_t _channels, uint32_t _columns, uint32_t _rows, uint32_t _frames, uint16_t _hash)
        : data(NULL), channels(_channels), columns(_columns), rows(_rows), frames(_frames), hash(_hash)
    {
        setSingleChannel(channels == 1);
        setSingleColumn(columns == 1);
        setSingleRow(rows == 1);
        setSingleFrame(frames == 1);
    }

    inline void copyHeader(const Matrix &other) { channels = other.channels; columns = other.columns; rows = other.rows; frames = other.frames; hash = other.hash; }
    inline void allocate() { deallocate(); data = new uint8_t[bytes()]; }
    inline void deallocate() { delete[] data; data = NULL; }*/
};

#ifdef __cplusplus
extern "C" {
#endif

BR_EXPORT int bits(const Matrix *m);
BR_EXPORT void setBits(Matrix *m, int bits);
BR_EXPORT bool isFloating(const Matrix *m);
BR_EXPORT void setFloating(Matrix *m, bool isFloating);
BR_EXPORT bool isSigned(const Matrix *m);
BR_EXPORT void setSigned(Matrix *m, bool isSigned);
BR_EXPORT int type(const Matrix *m);
BR_EXPORT void setType(Matrix *m, int type);
BR_EXPORT bool singleChannel(const Matrix *m);
BR_EXPORT void setSingleChannel(Matrix *m, bool singleChannel);
BR_EXPORT bool singleColumn(const Matrix *m);
BR_EXPORT void setSingleColumn(Matrix *m, bool singleColumn);
BR_EXPORT bool singleRow(const Matrix *m);
BR_EXPORT void setSingleRow(Matrix *m, bool singleRow);
BR_EXPORT bool singleFrame(const Matrix *m);
BR_EXPORT void setSingleFrame(Matrix *m, bool singleFrame);
BR_EXPORT uint32_t elements(const Matrix *m);
BR_EXPORT uint32_t bytes(const Matrix *m);

typedef void (*UnaryFunction)(const Matrix *src, Matrix *dst);
typedef void (*BinaryFunction)(const Matrix *srcA, const Matrix *srcB, Matrix *dst);
BR_EXPORT UnaryFunction makeUnaryFunction(const char *description);
BR_EXPORT BinaryFunction makeBinaryFunction(const char *description);

typedef uint32_t (*UnaryAllocation)(const Matrix *src, Matrix *dst);
typedef uint32_t (*BinaryAllocation)(const Matrix *srcA, const Matrix *srcB, Matrix *dst);
BR_EXPORT UnaryAllocation makeUnaryAllocation(const char *description, const Matrix *src);
BR_EXPORT BinaryAllocation makeBinaryAllocation(const char *description, const Matrix *srcA, const Matrix *srcB);

typedef void (*UnaryKernel)(const Matrix *src, Matrix *dst, uint32_t size);
typedef void (*BinaryKernel)(const Matrix *srcA, const Matrix *srcB, Matrix *dst, uint32_t size);
BR_EXPORT UnaryKernel makeUnaryKernel(const char *description, const Matrix *src);
BR_EXPORT BinaryKernel makeBinaryKernel(const char *description, const Matrix *srcA, const Matrix *srcB);

/*! @}*/

#ifdef __cplusplus
}
#endif

#endif // __LIKELY_H
