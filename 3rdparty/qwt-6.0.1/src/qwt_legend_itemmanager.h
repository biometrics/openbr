/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_LEGEND_ITEM_MANAGER_H
#define QWT_LEGEND_ITEM_MANAGER_H

#include "qwt_global.h"

class QwtLegend;
class QWidget;
class QRectF;
class QPainter;

/*!
  \brief Abstract API to bind plot items to the legend
*/

class QWT_EXPORT QwtLegendItemManager
{
public:
    //! Constructor
    QwtLegendItemManager()
    {
    }

    //! Destructor
    virtual ~QwtLegendItemManager()
    {
    }

    /*!
      Update the widget that represents the item on the legend
      \param legend Legend
      \sa legendItem()
     */
    virtual void updateLegend( QwtLegend *legend ) const = 0;

    /*!
      Allocate the widget that represents the item on the legend
      \return Allocated widget
      \sa updateLegend() QwtLegend()
     */

    virtual QWidget *legendItem() const = 0;

    /*!
      QwtLegendItem can display an icon-identifier followed
      by a text. The icon helps to identify a plot item on
      the plot canvas and depends on the type of information,
      that is displayed.

      The default implementation paints nothing.
     */
    virtual void drawLegendIdentifier( QPainter *, const QRectF & ) const
    {
    }
};

#endif

