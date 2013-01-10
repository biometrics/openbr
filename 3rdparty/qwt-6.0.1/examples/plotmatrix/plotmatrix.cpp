/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

// vim: expandtab

#include <qlayout.h>
#include <qpen.h>
#include <qwt_plot.h>
#include <qwt_scale_widget.h>
#include <qwt_scale_draw.h>
#include "plotmatrix.h"

class PlotMatrix::PrivateData
{
public:
    PrivateData():
        inScaleSync(false)
    {
        isAxisEnabled[QwtPlot::xBottom] = true;
        isAxisEnabled[QwtPlot::xTop] = false;
        isAxisEnabled[QwtPlot::yLeft] = true;
        isAxisEnabled[QwtPlot::yRight] = false;
    }

    bool isAxisEnabled[QwtPlot::axisCnt];
    QVector<QwtPlot *> plotWidgets;
    mutable bool inScaleSync;
};

PlotMatrix::PlotMatrix( int numRows, int numColumns, QWidget *parent):
    QFrame(parent)
{
    d_data = new PrivateData();
    d_data->plotWidgets.resize(numRows * numColumns);

    QGridLayout *layout = new QGridLayout(this);
    for ( int row = 0; row < numRows; row++ )
    {
        for ( int col = 0; col < numColumns; col++ )
        {
            QwtPlot *plot = new QwtPlot(this);
            layout->addWidget(plot, row, col);

            for ( int axis = 0; axis < QwtPlot::axisCnt; axis++ )
            {
                connect(plot->axisWidget(axis), 
                    SIGNAL(scaleDivChanged()), SLOT(scaleDivChanged()) );
            }
            d_data->plotWidgets[row * numColumns + col] = plot;
        }
    }

    updateLayout();
}

PlotMatrix::~PlotMatrix()
{
    delete d_data;
}

int PlotMatrix::numRows() const
{
    const QGridLayout *l = qobject_cast<const QGridLayout *>(layout());
    if ( l )
        return l->rowCount(); 

    return 0;
}

int PlotMatrix::numColumns() const
{
    const QGridLayout *l = qobject_cast<const QGridLayout *>(layout());
    if ( l )
        return l->columnCount(); 
    return 0;
}

QwtPlot* PlotMatrix::plot(int row, int column)
{
    const int index = row * numColumns() + column;
    if ( index < (int)d_data->plotWidgets.size() )
        return d_data->plotWidgets[index];

    return NULL;
}

const QwtPlot* PlotMatrix::plot(int row, int column) const
{
    const int index = row * numColumns() + column;
    if ( index < (int)d_data->plotWidgets.size() )
        return d_data->plotWidgets[index];

    return NULL;
}

void PlotMatrix::enableAxis(int axis, bool tf)
{
    if ( axis >= 0 && axis < QwtPlot::axisCnt )
    {
        if ( tf != d_data->isAxisEnabled[axis] )
        {
            d_data->isAxisEnabled[axis] = tf;
            updateLayout();
        }
    }
}

bool PlotMatrix::axisEnabled(int axis) const 
{
    if ( axis >= 0 && axis < QwtPlot::axisCnt )
        return d_data->isAxisEnabled[axis];

    return false;
}

void PlotMatrix::setAxisScale(int axis, int rowOrColumn,
    double min, double max, double step)
{
    int row = 0;
    int col = 0;

    if ( axis == QwtPlot::xBottom || axis == QwtPlot::xTop )
        col = rowOrColumn;
    else
        row = rowOrColumn;

    QwtPlot *plt = plot(row, col);
    if ( plt )
    {
        plt->setAxisScale(axis, min, max, step);
        plt->updateAxes();
    }
}

void PlotMatrix::scaleDivChanged()
{
    if ( d_data->inScaleSync )
        return;

    d_data->inScaleSync = true;

    QwtPlot *plt = NULL;
    int axisId = -1;
    int rowOrColumn = -1;

    // find the changed axis
    for ( int row = 0; row < numRows(); row++ )
    {
        for ( int col = 0; col < numColumns(); col++ )
        {
            QwtPlot *p = plot(row, col);
            if ( p )
            {
                for ( int axis = 0; axis < QwtPlot::axisCnt; axis++ )
                {
                    if ( p->axisWidget(axis) == sender() )
                    {
                        plt = p;
                        axisId = axis;
                        if ( axisId == QwtPlot::xBottom || axisId == QwtPlot::xTop )
                            rowOrColumn = col;
                        else
                            rowOrColumn = row;
                            
                    }
                }
            }
        }
    }

    if ( plt )
    {

        // synchronize the axes
        if ( axisId == QwtPlot::xBottom || axisId == QwtPlot::xTop )
        {
            for ( int row = 0; row < numRows(); row++ )
            {
                QwtPlot *p = plot(row, rowOrColumn);
                if ( p != plt )
                    p->setAxisScaleDiv(axisId, *plt->axisScaleDiv(axisId));
            }
        }
        else
        {
            for ( int col = 0; col < numColumns(); col++ )
            {
                QwtPlot *p = plot(rowOrColumn, col);
                if ( p != plt )
                    p->setAxisScaleDiv(axisId, *plt->axisScaleDiv(axisId));
            }
        }

        updateLayout();
    }

    d_data->inScaleSync = false;
}

void PlotMatrix::updateLayout()
{
    for ( int row = 0; row < numRows(); row++ )
    {
        for ( int col = 0; col < numColumns(); col++ )
        {
            QwtPlot *p = plot(row, col);
            if ( p )
            {
                bool showAxis[QwtPlot::axisCnt];
                showAxis[QwtPlot::xBottom] = 
                    axisEnabled(QwtPlot::xBottom) && row == numRows() - 1;
                showAxis[QwtPlot::xTop] = 
                    axisEnabled(QwtPlot::xTop) && row == 0;
                showAxis[QwtPlot::yLeft] = 
                    axisEnabled(QwtPlot::yLeft) && col == 0;
                showAxis[QwtPlot::yRight] = 
                    axisEnabled(QwtPlot::yRight) && col == numColumns() - 1;

                for ( int axis = 0; axis < QwtPlot::axisCnt; axis++ )
                {
                    if ( axis == QwtPlot::xBottom || axis == QwtPlot::xTop )
                        p->enableAxis(axis, showAxis[axis]);
                    else
                    {
                        p->enableAxis(axis, true);

                        QwtScaleDraw *sd = p->axisScaleDraw(axis);
                        sd->enableComponent(
                            QwtScaleDraw::Backbone, showAxis[axis]);
                        sd->enableComponent(
                            QwtScaleDraw::Ticks, showAxis[axis]);
                        sd->enableComponent(
                            QwtScaleDraw::Labels, showAxis[axis]);
                    }
                }
            }
        }
    }

    for ( int col = 0; col < numColumns(); col++ )
    {
        alignVAxes(col, QwtPlot::yLeft);
        alignVAxes(col, QwtPlot::yRight);
    }

    for ( int row = 0; row < numRows(); row++ )
    {
        for ( int col = 0; col < numColumns(); col++ )
        {
            QwtPlot *p = plot(row, col);
            if ( p )
                p->replot();
        }
    }
} 

void PlotMatrix::alignVAxes(int col, int axis)
{
    if ( axis != QwtPlot::yLeft && axis != QwtPlot::yRight )
        return;

    int maxExtent = 0;
    for ( int row = 0; row < numRows(); row++ )
    {
        QwtPlot *p = plot(row, col);
        if ( p )
        {
            QwtScaleWidget *scaleWidget = p->axisWidget(axis);

            QwtScaleDraw *sd = scaleWidget->scaleDraw();
            sd->setMinimumExtent(0);

            const int extent = sd->extent( scaleWidget->font() );
            if ( extent > maxExtent )
                maxExtent = extent;
        }
    }
    for ( int row = 0; row < numRows(); row++ )
    {
        QwtPlot *p = plot(row, col);
        if ( p )
        {
            QwtScaleWidget *scaleWidget = p->axisWidget(axis);
            scaleWidget->scaleDraw()->setMinimumExtent(maxExtent);
        }
    }
}
