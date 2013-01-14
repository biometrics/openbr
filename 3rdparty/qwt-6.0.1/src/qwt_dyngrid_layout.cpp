/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_dyngrid_layout.h"
#include "qwt_math.h"
#include <qwidget.h>
#include <qlist.h>

class QwtDynGridLayout::PrivateData
{
public:
    PrivateData():
        isDirty( true )
    {
    }

    void updateLayoutCache();

    mutable QList<QLayoutItem*> itemList;

    uint maxCols;
    uint numRows;
    uint numCols;

    Qt::Orientations expanding;

    bool isDirty;
    QVector<QSize> itemSizeHints;
};

void QwtDynGridLayout::PrivateData::updateLayoutCache()
{
    itemSizeHints.resize( itemList.count() );

    int index = 0;

    for ( QList<QLayoutItem*>::iterator it = itemList.begin();
        it != itemList.end(); ++it, index++ )
    {
        itemSizeHints[ index ] = ( *it )->sizeHint();
    }

    isDirty = false;
}

/*!
  \param parent Parent widget
  \param margin Margin
  \param spacing Spacing
*/

QwtDynGridLayout::QwtDynGridLayout( QWidget *parent,
        int margin, int spacing ):
    QLayout( parent )
{
    init();

    setSpacing( spacing );
    setMargin( margin );
}

/*!
  \param spacing Spacing
*/

QwtDynGridLayout::QwtDynGridLayout( int spacing )
{
    init();
    setSpacing( spacing );
}

/*!
  Initialize the layout with default values.
*/
void QwtDynGridLayout::init()
{
    d_data = new QwtDynGridLayout::PrivateData;
    d_data->maxCols = d_data->numRows = d_data->numCols = 0;
    d_data->expanding = 0;
}

//! Destructor

QwtDynGridLayout::~QwtDynGridLayout()
{
    for ( int i = 0; i < d_data->itemList.size(); i++ )
        delete d_data->itemList[i];

    delete d_data;
}

//! Invalidate all internal caches
void QwtDynGridLayout::invalidate()
{
    d_data->isDirty = true;
    QLayout::invalidate();
}

/*!
  Limit the number of columns.
  \param maxCols upper limit, 0 means unlimited
  \sa maxCols()
*/
void QwtDynGridLayout::setMaxCols( uint maxCols )
{
    d_data->maxCols = maxCols;
}

/*!
  Return the upper limit for the number of columns.
  0 means unlimited, what is the default.
  \sa setMaxCols()
*/

uint QwtDynGridLayout::maxCols() const
{
    return d_data->maxCols;
}

//! Adds item to the next free position.

void QwtDynGridLayout::addItem( QLayoutItem *item )
{
    d_data->itemList.append( item );
    invalidate();
}

/*!
  \return true if this layout is empty.
*/

bool QwtDynGridLayout::isEmpty() const
{
    return d_data->itemList.isEmpty();
}

/*!
  \return number of layout items
*/

uint QwtDynGridLayout::itemCount() const
{
    return d_data->itemList.count();
}

/*!
  Find the item at a spcific index

  \param index Index
  \sa takeAt()
*/
QLayoutItem *QwtDynGridLayout::itemAt( int index ) const
{
    if ( index < 0 || index >= d_data->itemList.count() )
        return NULL;

    return d_data->itemList.at( index );
}

/*!
  Find the item at a spcific index and remove it from the layout

  \param index Index
  \sa itemAt()
*/
QLayoutItem *QwtDynGridLayout::takeAt( int index )
{
    if ( index < 0 || index >= d_data->itemList.count() )
        return NULL;

    d_data->isDirty = true;
    return d_data->itemList.takeAt( index );
}

//! \return Number of items in the layout
int QwtDynGridLayout::count() const
{
    return d_data->itemList.count();
}

/*!
  Set whether this layout can make use of more space than sizeHint().
  A value of Qt::Vertical or Qt::Horizontal means that it wants to grow in only
  one dimension, while Qt::Vertical | Qt::Horizontal means that it wants
  to grow in both dimensions. The default value is 0.

  \param expanding Or'd orientations
  \sa expandingDirections()
*/
void QwtDynGridLayout::setExpandingDirections( Qt::Orientations expanding )
{
    d_data->expanding = expanding;
}

/*!
  Returns whether this layout can make use of more space than sizeHint().
  A value of Qt::Vertical or Qt::Horizontal means that it wants to grow in only
  one dimension, while Qt::Vertical | Qt::Horizontal means that it wants
  to grow in both dimensions.
  \sa setExpandingDirections()
*/
Qt::Orientations QwtDynGridLayout::expandingDirections() const
{
    return d_data->expanding;
}

/*!
  Reorganizes columns and rows and resizes managed widgets within
  the rectangle rect.

  \param rect Layout geometry
*/
void QwtDynGridLayout::setGeometry( const QRect &rect )
{
    QLayout::setGeometry( rect );

    if ( isEmpty() )
        return;

    d_data->numCols = columnsForWidth( rect.width() );
    d_data->numRows = itemCount() / d_data->numCols;
    if ( itemCount() % d_data->numCols )
        d_data->numRows++;

    QList<QRect> itemGeometries = layoutItems( rect, d_data->numCols );

    int index = 0;
    for ( QList<QLayoutItem*>::iterator it = d_data->itemList.begin();
        it != d_data->itemList.end(); ++it )
    {
        QWidget *w = ( *it )->widget();
        if ( w )
        {
            w->setGeometry( itemGeometries[index] );
            index++;
        }
    }
}

/*!
  Calculate the number of columns for a given width. It tries to
  use as many columns as possible (limited by maxCols())

  \param width Available width for all columns
  \sa maxCols(), setMaxCols()
*/

uint QwtDynGridLayout::columnsForWidth( int width ) const
{
    if ( isEmpty() )
        return 0;

    const int maxCols = ( d_data->maxCols > 0 ) ? d_data->maxCols : itemCount();
    if ( maxRowWidth( maxCols ) <= width )
        return maxCols;

    for ( int numCols = 2; numCols <= maxCols; numCols++ )
    {
        const int rowWidth = maxRowWidth( numCols );
        if ( rowWidth > width )
            return numCols - 1;
    }

    return 1; // At least 1 column
}

/*!
  Calculate the width of a layout for a given number of
  columns.

  \param numCols Given number of columns
  \param itemWidth Array of the width hints for all items
*/
int QwtDynGridLayout::maxRowWidth( int numCols ) const
{
    int col;

    QVector<int> colWidth( numCols );
    for ( col = 0; col < numCols; col++ )
        colWidth[col] = 0;

    if ( d_data->isDirty )
        d_data->updateLayoutCache();

    for ( int index = 0;
        index < d_data->itemSizeHints.count(); index++ )
    {
        col = index % numCols;
        colWidth[col] = qMax( colWidth[col],
            d_data->itemSizeHints[int( index )].width() );
    }

    int rowWidth = 2 * margin() + ( numCols - 1 ) * spacing();
    for ( col = 0; col < numCols; col++ )
        rowWidth += colWidth[col];

    return rowWidth;
}

/*!
  \return the maximum width of all layout items
*/
int QwtDynGridLayout::maxItemWidth() const
{
    if ( isEmpty() )
        return 0;

    if ( d_data->isDirty )
        d_data->updateLayoutCache();

    int w = 0;
    for ( int i = 0; i < d_data->itemSizeHints.count(); i++ )
    {
        const int itemW = d_data->itemSizeHints[i].width();
        if ( itemW > w )
            w = itemW;
    }

    return w;
}

/*!
  Calculate the geometries of the layout items for a layout
  with numCols columns and a given rect.

  \param rect Rect where to place the items
  \param numCols Number of columns
  \return item geometries
*/

QList<QRect> QwtDynGridLayout::layoutItems( const QRect &rect,
    uint numCols ) const
{
    QList<QRect> itemGeometries;
    if ( numCols == 0 || isEmpty() )
        return itemGeometries;

    uint numRows = itemCount() / numCols;
    if ( numRows % itemCount() )
        numRows++;

    QVector<int> rowHeight( numRows );
    QVector<int> colWidth( numCols );

    layoutGrid( numCols, rowHeight, colWidth );

    bool expandH, expandV;
    expandH = expandingDirections() & Qt::Horizontal;
    expandV = expandingDirections() & Qt::Vertical;

    if ( expandH || expandV )
        stretchGrid( rect, numCols, rowHeight, colWidth );

    const int maxCols = d_data->maxCols;
    d_data->maxCols = numCols;
    const QRect alignedRect = alignmentRect( rect );
    d_data->maxCols = maxCols;

    const int xOffset = expandH ? 0 : alignedRect.x();
    const int yOffset = expandV ? 0 : alignedRect.y();

    QVector<int> colX( numCols );
    QVector<int> rowY( numRows );

    const int xySpace = spacing();

    rowY[0] = yOffset + margin();
    for ( int r = 1; r < ( int )numRows; r++ )
        rowY[r] = rowY[r-1] + rowHeight[r-1] + xySpace;

    colX[0] = xOffset + margin();
    for ( int c = 1; c < ( int )numCols; c++ )
        colX[c] = colX[c-1] + colWidth[c-1] + xySpace;

    const int itemCount = d_data->itemList.size();
    for ( int i = 0; i < itemCount; i++ )
    {
        const int row = i / numCols;
        const int col = i % numCols;

        QRect itemGeometry( colX[col], rowY[row],
            colWidth[col], rowHeight[row] );
        itemGeometries.append( itemGeometry );
    }

    return itemGeometries;
}


/*!
  Calculate the dimensions for the columns and rows for a grid
  of numCols columns.

  \param numCols Number of columns.
  \param rowHeight Array where to fill in the calculated row heights.
  \param colWidth Array where to fill in the calculated column widths.
*/

void QwtDynGridLayout::layoutGrid( uint numCols,
    QVector<int>& rowHeight, QVector<int>& colWidth ) const
{
    if ( numCols <= 0 )
        return;

    if ( d_data->isDirty )
        d_data->updateLayoutCache();

    for ( uint index = 0;
        index < ( uint )d_data->itemSizeHints.count(); index++ )
    {
        const int row = index / numCols;
        const int col = index % numCols;

        const QSize &size = d_data->itemSizeHints[int( index )];

        rowHeight[row] = ( col == 0 )
            ? size.height() : qMax( rowHeight[row], size.height() );
        colWidth[col] = ( row == 0 )
            ? size.width() : qMax( colWidth[col], size.width() );
    }
}

/*!
  \return true: QwtDynGridLayout implements heightForWidth.
  \sa heightForWidth()
*/
bool QwtDynGridLayout::hasHeightForWidth() const
{
    return true;
}

/*!
  \return The preferred height for this layout, given the width w.
  \sa hasHeightForWidth()
*/
int QwtDynGridLayout::heightForWidth( int width ) const
{
    if ( isEmpty() )
        return 0;

    const uint numCols = columnsForWidth( width );
    uint numRows = itemCount() / numCols;
    if ( itemCount() % numCols )
        numRows++;

    QVector<int> rowHeight( numRows );
    QVector<int> colWidth( numCols );

    layoutGrid( numCols, rowHeight, colWidth );

    int h = 2 * margin() + ( numRows - 1 ) * spacing();
    for ( int row = 0; row < ( int )numRows; row++ )
        h += rowHeight[row];

    return h;
}

/*!
  Stretch columns in case of expanding() & QSizePolicy::Horizontal and
  rows in case of expanding() & QSizePolicy::Vertical to fill the entire
  rect. Rows and columns are stretched with the same factor.

  \sa setExpanding(), expanding()
*/
void QwtDynGridLayout::stretchGrid( const QRect &rect,
    uint numCols, QVector<int>& rowHeight, QVector<int>& colWidth ) const
{
    if ( numCols == 0 || isEmpty() )
        return;

    bool expandH, expandV;
    expandH = expandingDirections() & Qt::Horizontal;
    expandV = expandingDirections() & Qt::Vertical;

    if ( expandH )
    {
        int xDelta = rect.width() - 2 * margin() - ( numCols - 1 ) * spacing();
        for ( int col = 0; col < ( int )numCols; col++ )
            xDelta -= colWidth[col];

        if ( xDelta > 0 )
        {
            for ( int col = 0; col < ( int )numCols; col++ )
            {
                const int space = xDelta / ( numCols - col );
                colWidth[col] += space;
                xDelta -= space;
            }
        }
    }

    if ( expandV )
    {
        uint numRows = itemCount() / numCols;
        if ( itemCount() % numCols )
            numRows++;

        int yDelta = rect.height() - 2 * margin() - ( numRows - 1 ) * spacing();
        for ( int row = 0; row < ( int )numRows; row++ )
            yDelta -= rowHeight[row];

        if ( yDelta > 0 )
        {
            for ( int row = 0; row < ( int )numRows; row++ )
            {
                const int space = yDelta / ( numRows - row );
                rowHeight[row] += space;
                yDelta -= space;
            }
        }
    }
}

/*!
   Return the size hint. If maxCols() > 0 it is the size for
   a grid with maxCols() columns, otherwise it is the size for
   a grid with only one row.

   \sa maxCols(), setMaxCols()
*/
QSize QwtDynGridLayout::sizeHint() const
{
    if ( isEmpty() )
        return QSize();

    const uint numCols = ( d_data->maxCols > 0 ) ? d_data->maxCols : itemCount();
    uint numRows = itemCount() / numCols;
    if ( itemCount() % numCols )
        numRows++;

    QVector<int> rowHeight( numRows );
    QVector<int> colWidth( numCols );

    layoutGrid( numCols, rowHeight, colWidth );

    int h = 2 * margin() + ( numRows - 1 ) * spacing();
    for ( int row = 0; row < ( int )numRows; row++ )
        h += rowHeight[row];

    int w = 2 * margin() + ( numCols - 1 ) * spacing();
    for ( int col = 0; col < ( int )numCols; col++ )
        w += colWidth[col];

    return QSize( w, h );
}

/*!
  \return Number of rows of the current layout.
  \sa numCols()
  \warning The number of rows might change whenever the geometry changes
*/
uint QwtDynGridLayout::numRows() const
{
    return d_data->numRows;
}

/*!
  \return Number of columns of the current layout.
  \sa numRows()
  \warning The number of columns might change whenever the geometry changes
*/
uint QwtDynGridLayout::numCols() const
{
    return d_data->numCols;
}
