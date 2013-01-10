/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_curve_fitter.h"
#include "qwt_math.h"
#include "qwt_spline.h"
#include <qstack.h>
#include <qvector.h>

#if QT_VERSION < 0x040601
#define qFabs(x) ::fabs(x)
#endif

//! Constructor
QwtCurveFitter::QwtCurveFitter()
{
}

//! Destructor
QwtCurveFitter::~QwtCurveFitter()
{
}

class QwtSplineCurveFitter::PrivateData
{
public:
    PrivateData():
        fitMode( QwtSplineCurveFitter::Auto ),
        splineSize( 250 )
    {
    }

    QwtSpline spline;
    QwtSplineCurveFitter::FitMode fitMode;
    int splineSize;
};

//! Constructor
QwtSplineCurveFitter::QwtSplineCurveFitter()
{
    d_data = new PrivateData;
}

//! Destructor
QwtSplineCurveFitter::~QwtSplineCurveFitter()
{
    delete d_data;
}

/*!
  Select the algorithm used for building the spline

  \param mode Mode representing a spline algorithm
  \sa fitMode()
*/
void QwtSplineCurveFitter::setFitMode( FitMode mode )
{
    d_data->fitMode = mode;
}

/*!
  \return Mode representing a spline algorithm
  \sa setFitMode()
*/
QwtSplineCurveFitter::FitMode QwtSplineCurveFitter::fitMode() const
{
    return d_data->fitMode;
}

/*!
  Assign a spline

  \param spline Spline
  \sa spline()
*/
void QwtSplineCurveFitter::setSpline( const QwtSpline &spline )
{
    d_data->spline = spline;
    d_data->spline.reset();
}

/*!
  \return Spline
  \sa setSpline()
*/
const QwtSpline &QwtSplineCurveFitter::spline() const
{
    return d_data->spline;
}

/*!
  \return Spline
  \sa setSpline()
*/
QwtSpline &QwtSplineCurveFitter::spline()
{
    return d_data->spline;
}

/*!
   Assign a spline size ( has to be at least 10 points )

   \param splineSize Spline size
   \sa splineSize()
*/
void QwtSplineCurveFitter::setSplineSize( int splineSize )
{
    d_data->splineSize = qMax( splineSize, 10 );
}

/*!
  \return Spline size
  \sa setSplineSize()
*/
int QwtSplineCurveFitter::splineSize() const
{
    return d_data->splineSize;
}

/*!
  Find a curve which has the best fit to a series of data points

  \param points Series of data points
  \return Curve points
*/
QPolygonF QwtSplineCurveFitter::fitCurve( const QPolygonF &points ) const
{
    const int size = points.size();
    if ( size <= 2 )
        return points;

    FitMode fitMode = d_data->fitMode;
    if ( fitMode == Auto )
    {
        fitMode = Spline;

        const QPointF *p = points.data();
        for ( int i = 1; i < size; i++ )
        {
            if ( p[i].x() <= p[i-1].x() )
            {
                fitMode = ParametricSpline;
                break;
            }
        };
    }

    if ( fitMode == ParametricSpline )
        return fitParametric( points );
    else
        return fitSpline( points );
}

QPolygonF QwtSplineCurveFitter::fitSpline( const QPolygonF &points ) const
{
    d_data->spline.setPoints( points );
    if ( !d_data->spline.isValid() )
        return points;

    QPolygonF fittedPoints( d_data->splineSize );

    const double x1 = points[0].x();
    const double x2 = points[int( points.size() - 1 )].x();
    const double dx = x2 - x1;
    const double delta = dx / ( d_data->splineSize - 1 );

    for ( int i = 0; i < d_data->splineSize; i++ )
    {
        QPointF &p = fittedPoints[i];

        const double v = x1 + i * delta;
        const double sv = d_data->spline.value( v );

        p.setX( v );
        p.setY( sv );
    }
    d_data->spline.reset();

    return fittedPoints;
}

QPolygonF QwtSplineCurveFitter::fitParametric( const QPolygonF &points ) const
{
    int i;
    const int size = points.size();

    QPolygonF fittedPoints( d_data->splineSize );
    QPolygonF splinePointsX( size );
    QPolygonF splinePointsY( size );

    const QPointF *p = points.data();
    QPointF *spX = splinePointsX.data();
    QPointF *spY = splinePointsY.data();

    double param = 0.0;
    for ( i = 0; i < size; i++ )
    {
        const double x = p[i].x();
        const double y = p[i].y();
        if ( i > 0 )
        {
            const double delta = qSqrt( qwtSqr( x - spX[i-1].y() )
                      + qwtSqr( y - spY[i-1].y() ) );
            param += qMax( delta, 1.0 );
        }
        spX[i].setX( param );
        spX[i].setY( x );
        spY[i].setX( param );
        spY[i].setY( y );
    }

    d_data->spline.setPoints( splinePointsX );
    if ( !d_data->spline.isValid() )
        return points;

    const double deltaX =
        splinePointsX[size - 1].x() / ( d_data->splineSize - 1 );
    for ( i = 0; i < d_data->splineSize; i++ )
    {
        const double dtmp = i * deltaX;
        fittedPoints[i].setX( d_data->spline.value( dtmp ) );
    }

    d_data->spline.setPoints( splinePointsY );
    if ( !d_data->spline.isValid() )
        return points;

    const double deltaY =
        splinePointsY[size - 1].x() / ( d_data->splineSize - 1 );
    for ( i = 0; i < d_data->splineSize; i++ )
    {
        const double dtmp = i * deltaY;
        fittedPoints[i].setY( d_data->spline.value( dtmp ) );
    }

    return fittedPoints;
}

class QwtWeedingCurveFitter::PrivateData
{
public:
    PrivateData():
        tolerance( 1.0 )
    {
    }

    double tolerance;
};

class QwtWeedingCurveFitter::Line
{
public:
    Line( int i1 = 0, int i2 = 0 ):
        from( i1 ),
        to( i2 )
    {
    }

    int from;
    int to;
};

/*!
   Constructor

   \param tolerance Tolerance
   \sa setTolerance(), tolerance()
*/
QwtWeedingCurveFitter::QwtWeedingCurveFitter( double tolerance )
{
    d_data = new PrivateData;
    setTolerance( tolerance );
}

//! Destructor
QwtWeedingCurveFitter::~QwtWeedingCurveFitter()
{
    delete d_data;
}

/*!
 Assign the tolerance

 The tolerance is the maximum distance, that is accaptable
 between the original curve and the smoothed curve.

 Increasing the tolerance will reduce the number of the
 resulting points.

 \param tolerance Tolerance

 \sa tolerance()
*/
void QwtWeedingCurveFitter::setTolerance( double tolerance )
{
    d_data->tolerance = qMax( tolerance, 0.0 );
}

/*!
  \return Tolerance
  \sa setTolerance()
*/
double QwtWeedingCurveFitter::tolerance() const
{
    return d_data->tolerance;
}

/*!
  \param points Series of data points
  \return Curve points
*/
QPolygonF QwtWeedingCurveFitter::fitCurve( const QPolygonF &points ) const
{
    QStack<Line> stack;
    stack.reserve( 500 );

    const QPointF *p = points.data();
    const int nPoints = points.size();

    QVector<bool> usePoint( nPoints, false );

    double distToSegment;

    stack.push( Line( 0, nPoints - 1 ) );

    while ( !stack.isEmpty() )
    {
        const Line r = stack.pop();

        // initialize line segment
        const double vecX = p[r.to].x() - p[r.from].x();
        const double vecY = p[r.to].y() - p[r.from].y();

        const double vecLength = qSqrt( vecX * vecX + vecY * vecY );

        const double unitVecX = ( vecLength != 0.0 ) ? vecX / vecLength : 0.0;
        const double unitVecY = ( vecLength != 0.0 ) ? vecY / vecLength : 0.0;

        double maxDist = 0.0;
        int nVertexIndexMaxDistance = r.from + 1;
        for ( int i = r.from + 1; i < r.to; i++ )
        {
            //compare to anchor
            const double fromVecX = p[i].x() - p[r.from].x();
            const double fromVecY = p[i].y() - p[r.from].y();
            const double fromVecLength =
                qSqrt( fromVecX * fromVecX + fromVecY * fromVecY );

            if ( fromVecX * unitVecX + fromVecY * unitVecY < 0.0 )
            {
                distToSegment = fromVecLength;
            }
            if ( fromVecX * unitVecX + fromVecY * unitVecY < 0.0 )
            {
                distToSegment = fromVecLength;
            }
            else
            {
                const double toVecX = p[i].x() - p[r.to].x();
                const double toVecY = p[i].y() - p[r.to].y();
                const double toVecLength = qSqrt( toVecX * toVecX + toVecY * toVecY );
                const double s = toVecX * ( -unitVecX ) + toVecY * ( -unitVecY );
                if ( s < 0.0 )
                    distToSegment = toVecLength;
                else
                {
                    distToSegment = qSqrt( qFabs( toVecLength * toVecLength - s * s ) );
                }
            }

            if ( maxDist < distToSegment )
            {
                maxDist = distToSegment;
                nVertexIndexMaxDistance = i;
            }
        }
        if ( maxDist <= d_data->tolerance )
        {
            usePoint[r.from] = true;
            usePoint[r.to] = true;
        }
        else
        {
            stack.push( Line( r.from, nVertexIndexMaxDistance ) );
            stack.push( Line( nVertexIndexMaxDistance, r.to ) );
        }
    }

    int cnt = 0;

    QPolygonF stripped( nPoints );
    for ( int i = 0; i < nPoints; i++ )
    {
        if ( usePoint[i] )
            stripped[cnt++] = p[i];
    }
    stripped.resize( cnt );
    return stripped;
}
