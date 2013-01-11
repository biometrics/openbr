#ifndef __JITCV_H
#define __JITCV_H

#include <stddef.h>
#include <stdint.h>

namespace jitcv
{

/*!
 * \brief jitcv matrix
 * \author Josh Klontz \cite jklontz
 * \note Not part of the core SDK
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

    Matrix() : data(NULL), channels(0), columns(0), rows(0), frames(0), hash(0) {}

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
    inline void deallocate() { delete[] data; data = NULL; }

    inline int bits() const { return hash & Bits; }
    inline void setBits(int bits) { hash &= ~Bits; hash |= bits & Bits; }
    inline bool isFloating() const { return hash & Floating; }
    inline void setFloating(bool isFloating) { isFloating ? setSigned(true), hash |= Floating : hash &= ~Floating; }
    inline bool isSigned() const { return hash & Signed; }
    inline void setSigned(bool isSigned) { isSigned ? hash |= Signed : hash &= ~Signed; }
    inline int type() const { return hash & (Bits + Floating + Signed); }
    inline void setType(int type) { hash &= ~(Bits + Floating + Signed); hash |= type & (Bits + Floating + Signed); }
    inline bool singleChannel() const { return hash & SingleChannel; }
    inline void setSingleChannel(bool singleChannel) { singleChannel ? hash |= SingleChannel : hash &= ~SingleChannel; }
    inline bool singleColumn() const { return hash & SingleColumn; }
    inline void setSingleColumn(bool singleColumn) { singleColumn ? hash |= SingleColumn : hash &= ~SingleColumn; }
    inline bool singleRow() const { return hash & SingleRow; }
    inline void setSingleRow(bool singleRow) { singleRow ? hash |= SingleRow : hash &= ~SingleRow; }
    inline bool singleFrame() const { return hash & SingleFrame; }
    inline void setSingleFrame(bool singleFrame) { singleFrame ? hash |= SingleFrame : hash &= ~SingleFrame; }
    inline uint32_t elements() const { return channels * columns * rows * frames; }
    inline uint32_t bytes() const { return bits() / 8 * elements(); }
};

typedef void (*UnaryFunction_t)(const Matrix *src, Matrix *dst);
typedef void (*BinaryFunction_t)(const Matrix *srcA, const Matrix *srcB, Matrix *dst);
typedef void (*UnaryKernel_t)(const Matrix *src, Matrix *dst, uint32_t size);
typedef void (*BinaryKernel_t)(const Matrix *srcA, const Matrix *srcB, Matrix *dst, uint32_t size);

UnaryFunction_t jit_unary_make(const char *description);
BinaryFunction_t jit_binary_make(const char *description);

}

#endif // __JITCV_H
