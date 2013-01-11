/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_dial.h"
#include "qwt_dial_needle.h"
#include "qwt_math.h"
#include "qwt_scale_engine.h"
#include "qwt_scale_map.h"
#include "qwt_painter.h"
#include <qpainter.h>
#include <qbitmap.h>
#include <qpalette.h>
#include <qpixmap.h>
#include <qevent.h>
#include <qalgorithms.h>
#include <qmath.h>
#include <qstyle.h>
#include <qstyleoption.h>
#include <qapplication.h>

#if QT_VERSION < 0x040601
#define qAtan(x) ::atan(x)
#endif

class QwtDial::PrivateData
{
public:
    PrivateData():
        frameShadow( Sunken ),
        lineWidth( 0 ),
        mode( RotateNeedle ),
        direction( Clockwise ),
        origin( 90.0 ),
        minScaleArc( 0.0 ),
        maxScaleArc( 0.0 ),
        scaleDraw( 0 ),
        maxMajIntv( 36 ),
        maxMinIntv( 10 ),
        scaleStep( 0.0 ),
        needle( 0 )
    {
    }

    ~PrivateData()
    {
        delete scaleDraw;
        delete needle;
    }
    Shadow frameShadow;
    int lineWidth;

    QwtDial::Mode mode;
    QwtDial::Direction direction;

    double origin;
    double minScaleArc;
    double maxScaleArc;

    QwtDialScaleDraw *scaleDraw;
    int maxMajIntv;
    int maxMinIntv;
    double scaleStep;

    QwtDialNeedle *needle;

    static double previousDir;
};

double QwtDial::PrivateData::previousDir = -1.0;

/*!
  Constructor

  \param parent Parent dial widget
*/
QwtDialScaleDraw::QwtDialScaleDraw( QwtDial *parent ):
    d_parent( parent ),
    d_penWidth( 1.0 )
{
}

/*!
  Set the pen width used for painting the scale

  \param penWidth Pen width
  \sa penWidth(), QwtDial::drawScale()
*/

void QwtDialScaleDraw::setPenWidth( double penWidth )
{
    d_penWidth = qMax( penWidth, 0.0 );
}

/*!
  \return Pen width used for painting the scale
  \sa setPenWidth, QwtDial::drawScale()
*/
double QwtDialScaleDraw::penWidth() const
{
    return d_penWidth;
}

/*!
  Call QwtDial::scaleLabel of the parent dial widget.

  \param value Value to display

  \sa QwtDial::scaleLabel()
*/
QwtText QwtDialScaleDraw::label( double value ) const
{
    if ( d_parent == NULL )
        return QwtRoundScaleDraw::label( value );

    return d_parent->scaleLabel( value );
}

/*!
  \brief Constructor
  \param parent Parent widget

  Create a dial widget with no scale and no needle.
  The default origin is 90.0 with no valid value. It accepts
  mouse and keyboard inputs and has no step size. The default mode
  is QwtDial::RotateNeedle.
*/
QwtDial::QwtDial( QWidget* parent ):
    QwtAbstractSlider( Qt::Horizontal, parent )
{
    initDial();
}

void QwtDial::initDial()
{
    d_data = new PrivateData;

    setFocusPolicy( Qt::TabFocus );

    QPalette p = palette();
    for ( int i = 0; i < QPalette::NColorGroups; i++ )
    {
        const QPalette::ColorGroup cg = ( QPalette::ColorGroup )i;

        // Base: background color of the circle inside the frame.
        // WindowText: background color of the circle inside the scale

        p.setColor( cg, QPalette::WindowText,
            p.color( cg, QPalette::Base ) );
    }
    setPalette( p );

    d_data->scaleDraw = new QwtDialScaleDraw( this );
    d_data->scaleDraw->setRadius( 0 );

    setScaleArc( 0.0, 360.0 ); // scale as a full circle
    setRange( 0.0, 360.0, 1.0, 10 ); // degrees as deafult
}

//!  Destructor
QwtDial::~QwtDial()
{
    delete d_data;
}

/*!
  Sets the frame shadow value from the frame style.
  \param shadow Frame shadow
  \sa setLineWidth(), QFrame::setFrameShadow()
*/
void QwtDial::setFrameShadow( Shadow shadow )
{
    if ( shadow != d_data->frameShadow )
    {
        d_data->frameShadow = shadow;
        if ( lineWidth() > 0 )
            update();
    }
}

/*!
  \return Frame shadow
  /sa setFrameShadow(), lineWidth(), QFrame::frameShadow
*/
QwtDial::Shadow QwtDial::frameShadow() const
{
    return d_data->frameShadow;
}

/*!
  Sets the line width

  \param lineWidth Line width
  \sa setFrameShadow()
*/
void QwtDial::setLineWidth( int lineWidth )
{
    if ( lineWidth < 0 )
        lineWidth = 0;

    if ( d_data->lineWidth != lineWidth )
    {
        d_data->lineWidth = lineWidth;
        update();
    }
}

/*!
  \return Line width of the frame
  \sa setLineWidth(), frameShadow(), lineWidth()
*/
int QwtDial::lineWidth() const
{
    return d_data->lineWidth;
}

/*!
  \return bounding rect of the circle inside the frame
  \sa setLineWidth(), scaleInnerRect(), boundingRect()
*/
QRectF QwtDial::innerRect() const
{
    const double lw = lineWidth();
    return boundingRect().adjusted( lw, lw, -lw, -lw );
}

/*!
  \return bounding rect of the dial including the frame
  \sa setLineWidth(), scaleInnerRect(), innerRect()
*/
QRectF QwtDial::boundingRect() const
{
    const QRectF cr = contentsRect();

    const double dim = qMin( cr.width(), cr.height() );

    QRectF inner( 0, 0, dim, dim );
    inner.moveCenter( cr.center() );

    return inner;
}

/*!
  \return rect inside the scale
  \sa setLineWidth(), boundingRect(), innerRect()
*/
QRectF QwtDial::scaleInnerRect() const
{
    QRectF rect = innerRect();

    if ( d_data->scaleDraw )
    {
        double scaleDist = qCeil( d_data->scaleDraw->extent( font() ) );
        scaleDist++; // margin

        rect.adjust( scaleDist, scaleDist, -scaleDist, -scaleDist );
    }

    return rect;
}

/*!
  \brief Change the mode of the meter.
  \param mode New mode

  The value of the meter is indicated by the difference
  between north of the scale and the direction of the needle.
  In case of QwtDial::RotateNeedle north is pointing
  to the origin() and the needle is rotating, in case of
  QwtDial::RotateScale, the needle points to origin()
  and the scale is rotating.

  The default mode is QwtDial::RotateNeedle.

  \sa mode(), setValue(), setOrigin()
*/
void QwtDial::setMode( Mode mode )
{
    if ( mode != d_data->mode )
    {
        d_data->mode = mode;
        update();
    }
}

/*!
  \return mode of the dial.

  The value of the dial is indicated by the difference
  between the origin and the direction of the needle.
  In case of QwtDial::RotateNeedle the scale arc is fixed
  to the origin() and the needle is rotating, in case of
  QwtDial::RotateScale, the needle points to origin()
  and the scale is rotating.

  The default mode is QwtDial::RotateNeedle.

  \sa setMode(), origin(), setScaleArc(), value()
*/
QwtDial::Mode QwtDial::mode() const
{
    return d_data->mode;
}

/*!
    Sets whether it is possible to step the value from the highest value to
    the lowest value and vice versa to on.

    \param wrapping en/disables wrapping

    \sa wrapping(), QwtDoubleRange::periodic()
    \note The meaning of wrapping is like the wrapping property of QSpinBox,
          but not like it is used in QDial.
*/
void QwtDial::setWrapping( bool wrapping )
{
    setPeriodic( wrapping );
}

/*!
    wrapping() holds whether it is possible to step the value from the
    highest value to the lowest value and vice versa.

    \sa setWrapping(), QwtDoubleRange::setPeriodic()
    \note The meaning of wrapping is like the wrapping property of QSpinBox,
          but not like it is used in QDial.
*/
bool QwtDial::wrapping() const
{
    return periodic();
}

/*!
    Set the direction of the dial (clockwise/counterclockwise)

    \param direction Direction
    \sa direction()
*/
void QwtDial::setDirection( Direction direction )
{
    if ( direction != d_data->direction )
    {
        d_data->direction = direction;
        update();
    }
}

/*!
   \return Direction of the dial

   The default direction of a dial is QwtDial::Clockwise

   \sa setDirection()
*/
QwtDial::Direction QwtDial::direction() const
{
    return d_data->direction;
}

/*!
   Paint the dial
   \param event Paint event
*/
void QwtDial::paintEvent( QPaintEvent *event )
{
    QPainter painter( this );
    painter.setClipRegion( event->region() );

    QStyleOption opt;
    opt.init(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);

    painter.setRenderHint( QPainter::Antialiasing, true );

    painter.save();
    drawContents( &painter );
    painter.restore();

    painter.save();
    drawFrame( &painter );
    painter.restore();

    if ( hasFocus() )
        drawFocusIndicator( &painter );
}

/*!
  Draw a dotted round circle, if !isReadOnly()

  \param painter Painter
*/
void QwtDial::drawFocusIndicator( QPainter *painter ) const
{
    if ( !isReadOnly() )
    {
        QRectF focusRect = innerRect();

        const int margin = 2;
        focusRect.adjust( margin, margin, -margin, -margin );

        QColor color = palette().color( QPalette::Base );
        if ( color.isValid() )
        {
            const QColor gray( Qt::gray );

            int h, s, v;
            color.getHsv( &h, &s, &v );
            color = ( v > 128 ) ? gray.dark( 120 ) : gray.light( 120 );
        }
        else
            color = Qt::darkGray;

        painter->save();
        painter->setBrush( Qt::NoBrush );
        painter->setPen( QPen( color, 0, Qt::DotLine ) );
        painter->drawEllipse( focusRect );
        painter->restore();
    }
}

/*!
  Draw the frame around the dial

  \param painter Painter
  \sa lineWidth(), frameShadow()
*/
void QwtDial::drawFrame( QPainter *painter )
{
    if ( lineWidth() <= 0 )
        return;

    const double lw2 = 0.5 * lineWidth();

    QRectF r = boundingRect();
    r.adjust( lw2, lw2, -lw2, -lw2 );

    QPen pen;

    switch ( d_data->frameShadow )
    {
        case QwtDial::Raised:
        case QwtDial::Sunken:
        {
            QColor c1 = palette().color( QPalette::Light );
            QColor c2 = palette().color( QPalette::Dark );

            if ( d_data->frameShadow == QwtDial::Sunken )
                qSwap( c1, c2 );

            QLinearGradient gradient( r.topLeft(), r.bottomRight() );
            gradient.setColorAt( 0.0, c1 );
#if 0
            gradient.setColorAt( 0.3, c1 );
            gradient.setColorAt( 0.7, c2 );
#endif
            gradient.setColorAt( 1.0, c2 );

            pen = QPen( gradient, lineWidth() );
            break;
        }
        default: // Plain
        {
            pen = QPen( palette().brush( QPalette::Dark ), lineWidth() );
        }
    }

    painter->save();

    painter->setPen( pen );
    painter->setBrush( Qt::NoBrush );
    painter->drawEllipse( r );

    painter->restore();
}

/*!
  \brief Draw the contents inside the frame

  QPalette::Window is the background color outside of the frame.
  QPalette::Base is the background color inside the frame.
  QPalette::WindowText is the background color inside the scale.

  \param painter Painter
  \sa boundingRect(), innerRect(),
    scaleInnerRect(), QWidget::setPalette()
*/
void QwtDial::drawContents( QPainter *painter ) const
{
    if ( testAttribute( Qt::WA_NoSystemBackground ) ||
        palette().brush( QPalette::Base ) !=
            palette().brush( QPalette::Window ) )
    {
        const QRectF br = boundingRect();

        painter->save();
        painter->setPen( Qt::NoPen );
        painter->setBrush( palette().brush( QPalette::Base ) );
        painter->drawEllipse( br );
        painter->restore();
    }


    const QRectF insideScaleRect = scaleInnerRect();
    if ( palette().brush( QPalette::WindowText ) !=
            palette().brush( QPalette::Base ) )
    {
        painter->save();
        painter->setPen( Qt::NoPen );
        painter->setBrush( palette().brush( QPalette::WindowText ) );
        painter->drawEllipse( insideScaleRect );
        painter->restore();
    }

    const QPointF center = insideScaleRect.center();
    const double radius = 0.5 * insideScaleRect.width();

    double direction = d_data->origin;

    if ( isValid() )
    {
        direction = d_data->minScaleArc;
        if ( maxValue() > minValue() && 
            d_data->maxScaleArc > d_data->minScaleArc )
        {
            const double ratio =
                ( value() - minValue() ) / ( maxValue() - minValue() );
            direction += ratio * ( d_data->maxScaleArc - d_data->minScaleArc );
        }

        if ( d_data->direction == QwtDial::CounterClockwise )
            direction = d_data->maxScaleArc - ( direction - d_data->minScaleArc );

        direction += d_data->origin;
        if ( direction >= 360.0 )
            direction -= 360.0;
        else if ( direction < 0.0 )
            direction += 360.0;
    }

    double origin = d_data->origin;
    if ( mode() == RotateScale )
    {
        origin -= direction - d_data->origin;
        direction = d_data->origin;
    }

    painter->save();
    drawScale( painter, center, radius, origin,
        d_data->minScaleArc, d_data->maxScaleArc );
    painter->restore();

    painter->save();
    drawScaleContents( painter, center, radius );
    painter->restore();

    if ( isValid() )
    {
        QPalette::ColorGroup cg;
        if ( isEnabled() )
            cg = hasFocus() ? QPalette::Active : QPalette::Inactive;
        else
            cg = QPalette::Disabled;

        painter->save();
        drawNeedle( painter, center, radius, direction, cg );
        painter->restore();
    }
}

/*!
  Draw the needle

  \param painter Painter
  \param center Center of the dial
  \param radius Length for the needle
  \param direction Direction of the needle in degrees, counter clockwise
  \param cg ColorGroup
*/
void QwtDial::drawNeedle( QPainter *painter, const QPointF &center,
    double radius, double direction, QPalette::ColorGroup cg ) const
{
    if ( d_data->needle )
    {
        direction = 360.0 - direction; // counter clockwise
        d_data->needle->draw( painter, center, radius, direction, cg );
    }
}

/*!
  Draw the scale

  \param painter Painter
  \param center Center of the dial
  \param radius Radius of the scale
  \param origin Origin of the scale
  \param minArc Minimum of the arc
  \param maxArc Minimum of the arc

  \sa QwtRoundScaleDraw::setAngleRange()
*/
void QwtDial::drawScale( QPainter *painter, const QPointF &center,
    double radius, double origin, double minArc, double maxArc ) const
{
    if ( d_data->scaleDraw == NULL )
        return;

    origin -= 270.0; // hardcoded origin of QwtScaleDraw

    double angle = maxArc - minArc;
    if ( angle > 360.0 )
        angle = ::fmod( angle, 360.0 );

    minArc += origin;
    if ( minArc < -360.0 )
        minArc = ::fmod( minArc, 360.0 );

    maxArc = minArc + angle;
    if ( maxArc > 360.0 )
    {
        // QwtRoundScaleDraw::setAngleRange accepts only values
        // in the range [-360.0..360.0]
        minArc -= 360.0;
        maxArc -= 360.0;
    }

    if ( d_data->direction == QwtDial::CounterClockwise )
        qSwap( minArc, maxArc );

    painter->setFont( font() );

    d_data->scaleDraw->setAngleRange( minArc, maxArc );
    d_data->scaleDraw->setRadius( radius );
    d_data->scaleDraw->moveCenter( center );

    QPalette pal = palette();

    const QColor textColor = pal.color( QPalette::Text );
    pal.setColor( QPalette::WindowText, textColor ); //ticks, backbone

    painter->setPen( QPen( textColor, d_data->scaleDraw->penWidth() ) );

    painter->setBrush( Qt::red );
    d_data->scaleDraw->draw( painter, pal );
}

/*!
  Draw the contents inside the scale

  Paints nothing.

  \param painter Painter
  \param center Center of the contents circle
  \param radius Radius of the contents circle
*/

void QwtDial::drawScaleContents( QPainter *painter,
    const QPointF &center, double radius ) const
{
    Q_UNUSED(painter);
    Q_UNUSED(center);
    Q_UNUSED(radius);
}

/*!
  Set a needle for the dial

  Qwt is missing a set of good looking needles.
  Contributions are very welcome.

  \param needle Needle
  \warning The needle will be deleted, when a different needle is
    set or in ~QwtDial()
*/
void QwtDial::setNeedle( QwtDialNeedle *needle )
{
    if ( needle != d_data->needle )
    {
        if ( d_data->needle )
            delete d_data->needle;

        d_data->needle = needle;
        update();
    }
}

/*!
  \return needle
  \sa setNeedle()
*/
const QwtDialNeedle *QwtDial::needle() const
{
    return d_data->needle;
}

/*!
  \return needle
  \sa setNeedle()
*/
QwtDialNeedle *QwtDial::needle()
{
    return d_data->needle;
}

//! QwtDoubleRange update hook
void QwtDial::rangeChange()
{
    updateScale();
}

/*!
  Update the scale with the current attributes
  \sa setScale()
*/
void QwtDial::updateScale()
{
    if ( d_data->scaleDraw )
    {
        QwtLinearScaleEngine scaleEngine;

        const QwtScaleDiv scaleDiv = scaleEngine.divideScale(
            minValue(), maxValue(),
            d_data->maxMajIntv, d_data->maxMinIntv, d_data->scaleStep );

        d_data->scaleDraw->setTransformation( scaleEngine.transformation() );
        d_data->scaleDraw->setScaleDiv( scaleDiv );
    }
}

//! Return the scale draw
QwtDialScaleDraw *QwtDial::scaleDraw()
{
    return d_data->scaleDraw;
}

//! Return the scale draw
const QwtDialScaleDraw *QwtDial::scaleDraw() const
{
    return d_data->scaleDraw;
}

/*!
  Set an individual scale draw

  \param scaleDraw Scale draw
  \warning The previous scale draw is deleted
*/
void QwtDial::setScaleDraw( QwtDialScaleDraw *scaleDraw )
{
    if ( scaleDraw != d_data->scaleDraw )
    {
        if ( d_data->scaleDraw )
            delete d_data->scaleDraw;

        d_data->scaleDraw = scaleDraw;
        updateScale();
        update();
    }
}

/*!
  Change the intervals of the scale

  \param maxMajIntv Maximum for the number of major steps
  \param maxMinIntv Maximum number of minor steps
  \param step Step size

  \sa QwtScaleEngine::divideScale()
*/
void QwtDial::setScale( int maxMajIntv, int maxMinIntv, double step )
{
    d_data->maxMajIntv = maxMajIntv;
    d_data->maxMinIntv = maxMinIntv;
    d_data->scaleStep = step;

    updateScale();
}

/*!
  A wrapper method for accessing the scale draw.

  \param components Scale components
  \sa QwtAbstractScaleDraw::enableComponent()
*/
void QwtDial::setScaleComponents( 
    QwtAbstractScaleDraw::ScaleComponents components )
{
    if ( components == 0 )
        setScaleDraw( NULL );

    QwtDialScaleDraw *sd = d_data->scaleDraw;
    if ( sd == NULL )
        return;

    sd->enableComponent( QwtAbstractScaleDraw::Backbone,
        components & QwtAbstractScaleDraw::Backbone );

    sd->enableComponent( QwtAbstractScaleDraw::Ticks,
        components & QwtAbstractScaleDraw::Ticks );

    sd->enableComponent( QwtAbstractScaleDraw::Labels,
        components & QwtAbstractScaleDraw::Labels );
}

/*!
  Assign length and width of the ticks

  \param minLen Length of the minor ticks
  \param medLen Length of the medium ticks
  \param majLen Length of the major ticks
  \param penWidth Width of the pen for all ticks

  \sa QwtAbstractScaleDraw::setTickLength(), QwtDialScaleDraw::setPenWidth()
*/
void QwtDial::setScaleTicks( int minLen, int medLen,
                             int majLen, int penWidth )
{
    QwtDialScaleDraw *sd = d_data->scaleDraw;
    if ( sd )
    {
        sd->setTickLength( QwtScaleDiv::MinorTick, minLen );
        sd->setTickLength( QwtScaleDiv::MediumTick, medLen );
        sd->setTickLength( QwtScaleDiv::MajorTick, majLen );
        sd->setPenWidth( penWidth );
    }
}

/*!
   Find the label for a value

   \param value Value
   \return label
*/
QwtText QwtDial::scaleLabel( double value ) const
{
#if 1
    if ( value == -0 )
        value = 0;
#endif

    return QString::number( value );
}

//! \return Lower limit of the scale arc
double QwtDial::minScaleArc() const
{
    return d_data->minScaleArc;
}

//! \return Upper limit of the scale arc
double QwtDial::maxScaleArc() const
{
    return d_data->maxScaleArc;
}

/*!
  \brief Change the origin

  The origin is the angle where scale and needle is relative to.

  \param origin New origin
  \sa origin()
*/
void QwtDial::setOrigin( double origin )
{
    d_data->origin = origin;
    update();
}

/*!
  The origin is the angle where scale and needle is relative to.

  \return Origin of the dial
  \sa setOrigin()
*/
double QwtDial::origin() const
{
    return d_data->origin;
}

/*!
  Change the arc of the scale

  \param minArc Lower limit
  \param maxArc Upper limit
*/
void QwtDial::setScaleArc( double minArc, double maxArc )
{
    if ( minArc != 360.0 && minArc != -360.0 )
        minArc = ::fmod( minArc, 360.0 );
    if ( maxArc != 360.0 && maxArc != -360.0 )
        maxArc = ::fmod( maxArc, 360.0 );

    d_data->minScaleArc = qMin( minArc, maxArc );
    d_data->maxScaleArc = qMax( minArc, maxArc );
    if ( d_data->maxScaleArc - d_data->minScaleArc > 360.0 )
        d_data->maxScaleArc = d_data->minScaleArc + 360.0;

    update();
}

//! QwtDoubleRange update hook
void QwtDial::valueChange()
{
    update();
    QwtAbstractSlider::valueChange();
}

/*!
  \return Size hint
*/
QSize QwtDial::sizeHint() const
{
    int sh = 0;
    if ( d_data->scaleDraw )
        sh = qCeil( d_data->scaleDraw->extent( font() ) );

    const int d = 6 * sh + 2 * lineWidth();

    QSize hint( d, d ); 
    if ( !isReadOnly() )
        hint = hint.expandedTo( QApplication::globalStrut() );

    return hint;
}

/*!
  \brief Return a minimum size hint
  \warning The return value of QwtDial::minimumSizeHint() depends on the
           font and the scale.
*/
QSize QwtDial::minimumSizeHint() const
{
    int sh = 0;
    if ( d_data->scaleDraw )
        sh = qCeil( d_data->scaleDraw->extent( font() ) );

    const int d = 3 * sh + 2 * lineWidth();

    return QSize( d, d );
}

static double line2Radians( const QPointF &p1, const QPointF &p2 )
{
    const QPointF p = p2 - p1;

    double angle;
    if ( p.x() == 0 )
        angle = ( p.y() <= 0.0 ) ? M_PI_2 : 3 * M_PI_2;
    else
    {
        angle = qAtan( double( -p.y() ) / double( p.x() ) );
        if ( p.x() < 0.0 )
            angle += M_PI;
        if ( angle < 0.0 )
            angle += 2 * M_PI;
    }
    return 360.0 - angle * 180.0 / M_PI;
}

/*!
  Find the value for a given position

  \param pos Position
  \return Value
*/
double QwtDial::getValue( const QPoint &pos )
{
    if ( d_data->maxScaleArc == d_data->minScaleArc || maxValue() == minValue() )
        return minValue();

    double dir = line2Radians( innerRect().center(), pos ) - d_data->origin;
    if ( dir < 0.0 )
        dir += 360.0;

    if ( mode() == RotateScale )
        dir = 360.0 - dir;

    // The position might be in the area that is outside the scale arc.
    // We need the range of the scale if it was a complete circle.

    const double completeCircle = 360.0 / ( d_data->maxScaleArc - d_data->minScaleArc )
        * ( maxValue() - minValue() );

    double posValue = minValue() + completeCircle * dir / 360.0;

    if ( scrollMode() == ScrMouse )
    {
        if ( d_data->previousDir >= 0.0 ) // valid direction
        {
            // We have to find out whether the mouse is moving
            // clock or counter clockwise

            bool clockWise = false;

            const double angle = dir - d_data->previousDir;
            if ( ( angle >= 0.0 && angle <= 180.0 ) || angle < -180.0 )
                clockWise = true;

            if ( clockWise )
            {
                if ( dir < d_data->previousDir && mouseOffset() > 0.0 )
                {
                    // We passed 360 -> 0
                    setMouseOffset( mouseOffset() - completeCircle );
                }

                if ( wrapping() )
                {
                    if ( posValue - mouseOffset() > maxValue() )
                    {
                        // We passed maxValue and the value will be set
                        // to minValue. We have to adjust the mouseOffset.

                        setMouseOffset( posValue - minValue() );
                    }
                }
                else
                {
                    if ( posValue - mouseOffset() > maxValue() ||
                            value() == maxValue() )
                    {
                        // We fix the value at maxValue by adjusting
                        // the mouse offset.

                        setMouseOffset( posValue - maxValue() );
                    }
                }
            }
            else
            {
                if ( dir > d_data->previousDir && mouseOffset() < 0.0 )
                {
                    // We passed 0 -> 360
                    setMouseOffset( mouseOffset() + completeCircle );
                }

                if ( wrapping() )
                {
                    if ( posValue - mouseOffset() < minValue() )
                    {
                        // We passed minValue and the value will be set
                        // to maxValue. We have to adjust the mouseOffset.

                        setMouseOffset( posValue - maxValue() );
                    }
                }
                else
                {
                    if ( posValue - mouseOffset() < minValue() ||
                        value() == minValue() )
                    {
                        // We fix the value at minValue by adjusting
                        // the mouse offset.

                        setMouseOffset( posValue - minValue() );
                    }
                }
            }
        }
        d_data->previousDir = dir;
    }

    return posValue;
}

/*!
  See QwtAbstractSlider::getScrollMode()

  \param pos point where the mouse was pressed
  \retval scrollMode The scrolling mode
  \retval direction  direction: 1, 0, or -1.

  \sa QwtAbstractSlider::getScrollMode()
*/
void QwtDial::getScrollMode( const QPoint &pos, 
    QwtAbstractSlider::ScrollMode &scrollMode, int &direction ) const
{
    direction = 0;
    scrollMode = QwtAbstractSlider::ScrNone;

    const QRegion region( innerRect().toRect(), QRegion::Ellipse );
    if ( region.contains( pos ) && pos != innerRect().center() )
    {
        scrollMode = QwtAbstractSlider::ScrMouse;
        d_data->previousDir = -1.0;
    }
}

/*!
  Handles key events

  - Key_Down, KeyLeft\n
    Decrement by 1
  - Key_Prior\n
    Decrement by pageSize()
  - Key_Home\n
    Set the value to minValue()

  - Key_Up, KeyRight\n
    Increment by 1
  - Key_Next\n
    Increment by pageSize()
  - Key_End\n
    Set the value to maxValue()

  \param event Key event
  \sa isReadOnly()
*/
void QwtDial::keyPressEvent( QKeyEvent *event )
{
    if ( isReadOnly() )
    {
        event->ignore();
        return;
    }

    if ( !isValid() )
        return;

    double previous = prevValue();
    switch ( event->key() )
    {
        case Qt::Key_Down:
        case Qt::Key_Left:
            QwtDoubleRange::incValue( -1 );
            break;
        case Qt::Key_PageUp:
            QwtDoubleRange::incValue( -pageSize() );
            break;
        case Qt::Key_Home:
            setValue( minValue() );
            break;

        case Qt::Key_Up:
        case Qt::Key_Right:
            QwtDoubleRange::incValue( 1 );
            break;
        case Qt::Key_PageDown:
            QwtDoubleRange::incValue( pageSize() );
            break;
        case Qt::Key_End:
            setValue( maxValue() );
            break;
        default:;
            event->ignore();
    }

    if ( value() != previous )
        Q_EMIT sliderMoved( value() );
}
