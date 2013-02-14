#ifndef __LIKELY_H
#define __LIKELY_H

#include <stddef.h>
#include <stdint.h>

#if defined BR_LIBRARY
#  if defined _WIN32 || defined __CYGWIN__
#    define BR_EXPORT __declspec(dllexport)
#  else
#    define BR_EXPORT __attribute__((visibility("default")))
#  endif
#else
#  if defined _WIN32 || defined __CYGWIN__
#    define BR_EXPORT __declspec(dllimport)
#  else
#    define BR_EXPORT
#  endif
#endif

/*!
 * \defgroup likely Literate Kernel Library
 * \brief Experimental low-level API for programming image processing kernels on heterogeneous hardware architectures.
 * \author Josh Klontz \cite jklontz
 */

/*!
 * \addtogroup likely
 *  @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Likely Matrix
 */
struct likely_matrix
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
};

BR_EXPORT int  likely_bits(const likely_matrix *m);
BR_EXPORT void likely_set_bits(likely_matrix *m, int bits);
BR_EXPORT bool likely_is_floating(const likely_matrix *m);
BR_EXPORT void likely_set_floating(likely_matrix *m, bool is_floating);
BR_EXPORT bool likely_is_signed(const likely_matrix *m);
BR_EXPORT void likely_set_signed(likely_matrix *m, bool is_signed);
BR_EXPORT int  likely_type(const likely_matrix *m);
BR_EXPORT void likely_set_type(likely_matrix *m, int type);
BR_EXPORT bool likely_is_single_channel(const likely_matrix *m);
BR_EXPORT void likely_set_single_channel(likely_matrix *m, bool is_single_channel);
BR_EXPORT bool likely_is_single_column(const likely_matrix *m);
BR_EXPORT void likely_set_single_column(likely_matrix *m, bool is_single_column);
BR_EXPORT bool likely_is_single_row(const likely_matrix *m);
BR_EXPORT void likely_set_single_row(likely_matrix *m, bool is_single_row);
BR_EXPORT bool likely_is_single_frame(const likely_matrix *m);
BR_EXPORT void likely_set_single_frame(likely_matrix *m, bool is_single_frame);
BR_EXPORT uint32_t likely_elements(const likely_matrix *m);
BR_EXPORT uint32_t likely_bytes(const likely_matrix *m);

typedef void (*likely_unary_function)(const likely_matrix *src, likely_matrix *dst);
typedef void (*likely_binary_function)(const likely_matrix *srcA, const likely_matrix *srcB, likely_matrix *dst);
BR_EXPORT likely_unary_function likely_make_unary_function(const char *description);
BR_EXPORT likely_binary_function likely_make_binary_function(const char *description);

typedef uint32_t (*likely_unary_allocation)(const likely_matrix *src, likely_matrix *dst);
typedef uint32_t (*likely_binary_allocation)(const likely_matrix *srcA, const likely_matrix *srcB, likely_matrix *dst);
BR_EXPORT likely_unary_allocation likely_make_unary_allocation(const char *description, const likely_matrix *src);
BR_EXPORT likely_binary_allocation likely_make_binary_allocation(const char *description, const likely_matrix *src_a, const likely_matrix *src_b);

typedef void (*likely_unary_kernel)(const likely_matrix *src, likely_matrix *dst, uint32_t size);
typedef void (*likely_binary_kernel)(const likely_matrix *srcA, const likely_matrix *srcB, likely_matrix *dst, uint32_t size);
BR_EXPORT likely_unary_kernel likely_make_unary_kernel(const char *description, const likely_matrix *src);
BR_EXPORT likely_binary_kernel likely_make_binary_kernel(const char *description, const likely_matrix *src_a, const likely_matrix *src_b);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace likely {

struct Matrix : public likely_matrix
{
    Matrix()
    {
        data = NULL;
        channels = columns = rows = frames = hash = 0;
    }

    Matrix(uint32_t _channels, uint32_t _columns, uint32_t _rows, uint32_t _frames, uint16_t _hash)
    {
        data = NULL;
        channels = _channels;
        columns = _columns;
        rows = _rows;
        frames = _frames;
        hash = _hash;
        setSingleChannel(channels == 1);
        setSingleColumn(columns == 1);
        setSingleRow(rows == 1);
        setSingleFrame(frames == 1);
    }

    inline int  bits() const { return likely_bits(this); }
    inline void setBits(int bits) { likely_set_bits(this, bits); }
    inline bool isFloating() const { return likely_is_floating(this); }
    inline void setFloating(bool isFloating) { likely_set_floating(this, isFloating); }
    inline bool isSigned() const { return likely_is_signed(this); }
    inline void setSigned(bool isSigned) { likely_set_signed(this, isSigned); }
    inline int  type() const { return likely_type(this); }
    inline void setType(int type) { likely_set_type(this, type); }
    inline bool isSingleChannel() const { return likely_is_single_channel(this); }
    inline void setSingleChannel(bool isSingleChannel) { likely_set_single_channel(this, isSingleChannel); }
    inline bool isSingleColumn() const { return likely_is_single_column(this); }
    inline void setSingleColumn(bool isSingleColumn) { likely_set_single_column(this, isSingleColumn); }
    inline bool isSingleRow() const { return likely_is_single_row(this); }
    inline void setSingleRow(bool isSingleRow) { likely_set_single_row(this, isSingleRow); }
    inline bool isSingleFrame() const { return likely_is_single_frame(this); }
    inline void setSingleFrame(bool isSingleFrame) { likely_set_single_frame(this, isSingleFrame); }
    inline uint32_t elements() const { return likely_elements(this); }
    inline uint32_t bytes() const { return likely_bytes(this); }
};

typedef likely_unary_function UnaryFunction;
typedef likely_binary_function BinaryFunction;
inline UnaryFunction makeUnaryFunction(const char *description) { return likely_make_unary_function(description); }
inline BinaryFunction makeBinaryFunction(const char *description) { return likely_make_binary_function(description); }

typedef likely_unary_allocation UnaryAllocation;
typedef likely_binary_allocation BinaryAllocation;
inline UnaryAllocation makeUnaryAllocation(const char *description, const Matrix &src) { return likely_make_unary_allocation(description, &src); }
inline BinaryAllocation makeBinaryAllocation(const char *description, const Matrix &srcA, const Matrix &srcB) { return likely_make_binary_allocation(description, &srcA, &srcB); }

typedef likely_unary_kernel UnaryKernel;
typedef likely_binary_kernel BinaryKernel;
inline UnaryKernel makeUnaryKernel(const char *description, const Matrix &src) { return likely_make_unary_kernel(description, &src); }
inline BinaryKernel makeBinaryKernel(const char *description, const Matrix &srcA, const Matrix &srcB) { return likely_make_binary_kernel(description, &srcA, &srcB); }

} // namespace likely

#endif // __cplusplus

/*! @}*/

#endif // __LIKELY_H
