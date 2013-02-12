#include "likely.h"

int bits(const Matrix *m)
{
    return m->hash & Matrix::Bits;
}

void setBits(Matrix *m, int bits)
{
    m->hash &= ~Matrix::Bits; m->hash |= bits & Matrix::Bits;
}

bool isFloating(const Matrix *m)
{
    return m->hash & Matrix::Floating;
}

void setFloating(Matrix *m, bool isFloating)
{
    isFloating ? setSigned(m, true), m->hash |= Matrix::Floating : m->hash &= ~Matrix::Floating;
}

bool isSigned(const Matrix *m)
{
    return m->hash & Matrix::Signed;
}

void setSigned(Matrix *m, bool isSigned)
{
    isSigned ? m->hash |= Matrix::Signed : m->hash &= ~Matrix::Signed;
}

int type(const Matrix *m)
{
    return m->hash & (Matrix::Bits + Matrix::Floating + Matrix::Signed);
}

void setType(Matrix *m, int type)
{
    m->hash &= ~(Matrix::Bits + Matrix::Floating + Matrix::Signed);
    m->hash |= type & (Matrix::Bits + Matrix::Floating + Matrix::Signed);
}

bool singleChannel(const Matrix *m)
{
    return m->hash & Matrix::SingleChannel;
}

void setSingleChannel(Matrix *m, bool singleChannel)
{
    singleChannel ? m->hash |= Matrix::SingleChannel : m->hash &= ~Matrix::SingleChannel;
}

bool singleColumn(const Matrix *m)
{
    return m->hash & Matrix::SingleColumn;
}

void setSingleColumn(Matrix *m, bool singleColumn)
{
    singleColumn ? m->hash |= Matrix::SingleColumn : m->hash &= ~Matrix::SingleColumn;
}

bool singleRow(const Matrix *m)
{
    return m->hash & Matrix::SingleRow;
}

void setSingleRow(Matrix *m, bool singleRow)
{
    singleRow ? m->hash |= Matrix::SingleRow : m->hash &= ~Matrix::SingleRow;
}

bool singleFrame(const Matrix *m)
{
    return m->hash & Matrix::SingleFrame;
}

void setSingleFrame(Matrix *m, bool singleFrame)
{
    singleFrame ? m->hash |= Matrix::SingleFrame : m->hash &= ~Matrix::SingleFrame;
}

uint32_t elements(const Matrix *m)
{
    return m->channels * m->columns * m->rows * m->frames;
}

uint32_t bytes(const Matrix *m)
{
    return bits(m) / 8 * elements(m);
}
