#include "likely.h"

int likely_bits(const likely_matrix *m)
{
    return m->hash & likely_matrix::Bits;
}

void likely_set_bits(likely_matrix *m, int bits)
{
    m->hash &= ~likely_matrix::Bits; m->hash |= bits & likely_matrix::Bits;
}

bool likely_is_floating(const likely_matrix *m)
{
    return m->hash & likely_matrix::Floating;
}

void likely_set_floating(likely_matrix *m, bool is_floating)
{
    is_floating ? likely_set_signed(m, true), m->hash |= likely_matrix::Floating : m->hash &= ~likely_matrix::Floating;
}

bool likely_is_signed(const likely_matrix *m)
{
    return m->hash & likely_matrix::Signed;
}

void likely_set_signed(likely_matrix *m, bool is_signed)
{
    is_signed ? m->hash |= likely_matrix::Signed : m->hash &= ~likely_matrix::Signed;
}

int likely_type(const likely_matrix *m)
{
    return m->hash & (likely_matrix::Bits + likely_matrix::Floating + likely_matrix::Signed);
}

void likely_set_type(likely_matrix *m, int type)
{
    m->hash &= ~(likely_matrix::Bits + likely_matrix::Floating + likely_matrix::Signed);
    m->hash |= type & (likely_matrix::Bits + likely_matrix::Floating + likely_matrix::Signed);
}

bool likely_is_single_channel(const likely_matrix *m)
{
    return m->hash & likely_matrix::SingleChannel;
}

void likely_set_single_channel(likely_matrix *m, bool is_single_channel)
{
    is_single_channel ? m->hash |= likely_matrix::SingleChannel : m->hash &= ~likely_matrix::SingleChannel;
}

bool likely_is_single_column(const likely_matrix *m)
{
    return m->hash & likely_matrix::SingleColumn;
}

void likely_set_single_column(likely_matrix *m, bool is_single_column)
{
    is_single_column ? m->hash |= likely_matrix::SingleColumn : m->hash &= ~likely_matrix::SingleColumn;
}

bool likely_is_single_row(const likely_matrix *m)
{
    return m->hash & likely_matrix::SingleRow;
}

void likely_set_single_row(likely_matrix *m, bool is_single_row)
{
    is_single_row ? m->hash |= likely_matrix::SingleRow : m->hash &= ~likely_matrix::SingleRow;
}

bool likely_is_single_frame(const likely_matrix *m)
{
    return m->hash & likely_matrix::SingleFrame;
}

void likely_set_single_frame(likely_matrix *m, bool is_single_frame)
{
    is_single_frame ? m->hash |= likely_matrix::SingleFrame : m->hash &= ~likely_matrix::SingleFrame;
}

uint32_t likely_elements(const likely_matrix *m)
{
    return m->channels * m->columns * m->rows * m->frames;
}

uint32_t likely_bytes(const likely_matrix *m)
{
    return likely_bits(m) / 8 * likely_elements(m);
}
