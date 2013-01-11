/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_arrow_button.h"
#include "qwt_math.h"
#include "qwt_counter.h"
#include <qlayout.h>
#include <qlineedit.h>
#include <qvalidator.h>
#include <qevent.h>
#include <qstyle.h>

class QwtCounter::PrivateData
{
public:
    PrivateData():
        editable( true )
    {
        increment[Button1] = 1;
        increment[Button2] = 10;
        increment[Button3] = 100;
    }

    QwtArrowButton *buttonDown[ButtonCnt];
    QwtArrowButton *buttonUp[ButtonCnt];
    QLineEdit *valueEdit;

    int increment[ButtonCnt];
    int numButtons;

    bool editable;
};

/*!
  The default number of buttons is set to 2. The default increments are:
  \li Button 1: 1 step
  \li Button 2: 10 steps
  \li Button 3: 100 steps

  \param parent
 */
QwtCounter::QwtCounter( QWidget *parent ):
    QWidget( parent )
{
    initCounter();
}

void QwtCounter::initCounter()
{
    d_data = new PrivateData;

    QHBoxLayout *layout = new QHBoxLayout( this );
    layout->setSpacing( 0 );
    layout->setMargin( 0 );

    for ( int i = ButtonCnt - 1; i >= 0; i-- )
    {
        QwtArrowButton *btn =
            new QwtArrowButton( i + 1, Qt::DownArrow, this );
        btn->setFocusPolicy( Qt::NoFocus );
        btn->installEventFilter( this );
        layout->addWidget( btn );

        connect( btn, SIGNAL( released() ), SLOT( btnReleased() ) );
        connect( btn, SIGNAL( clicked() ), SLOT( btnClicked() ) );

        d_data->buttonDown[i] = btn;
    }

    d_data->valueEdit = new QLineEdit( this );
    d_data->valueEdit->setReadOnly( false );
    d_data->valueEdit->setValidator( new QDoubleValidator( d_data->valueEdit ) );
    layout->addWidget( d_data->valueEdit );

    connect( d_data->valueEdit, SIGNAL( editingFinished() ),
         SLOT( textChanged() ) );

    layout->setStretchFactor( d_data->valueEdit, 10 );

    for ( int i = 0; i < ButtonCnt; i++ )
    {
        QwtArrowButton *btn =
            new QwtArrowButton( i + 1, Qt::UpArrow, this );
        btn->setFocusPolicy( Qt::NoFocus );
        btn->installEventFilter( this );
        layout->addWidget( btn );

        connect( btn, SIGNAL( released() ), SLOT( btnReleased() ) );
        connect( btn, SIGNAL( clicked() ), SLOT( btnClicked() ) );

        d_data->buttonUp[i] = btn;
    }

    setNumButtons( 2 );
    setRange( 0.0, 1.0, 0.001 );
    setValue( 0.0 );

    setSizePolicy(
        QSizePolicy( QSizePolicy::Preferred, QSizePolicy::Fixed ) );

    setFocusProxy( d_data->valueEdit );
    setFocusPolicy( Qt::StrongFocus );
}

//! Destructor
QwtCounter::~QwtCounter()
{
    delete d_data;
}

//! Set from lineedit
void QwtCounter::textChanged()
{
    if ( !d_data->editable )
        return;

    bool converted = false;

    const double value = d_data->valueEdit->text().toDouble( &converted );
    if ( converted )
        setValue( value );
}

/**
  \brief Allow/disallow the user to manually edit the value

  \param editable true enables editing
  \sa editable()
*/
void QwtCounter::setEditable( bool editable )
{
    if ( editable == d_data->editable )
        return;

    d_data->editable = editable;
    d_data->valueEdit->setReadOnly( !editable );
}

//! returns whether the line edit is edatble. (default is yes)
bool QwtCounter::editable() const
{
    return d_data->editable;
}

/*!
   Handle PolishRequest events
   \param event Event
*/
bool QwtCounter::event( QEvent *event )
{
    if ( event->type() == QEvent::PolishRequest )
    {
        const int w = d_data->valueEdit->fontMetrics().width( "W" ) + 8;
        for ( int i = 0; i < ButtonCnt; i++ )
        {
            d_data->buttonDown[i]->setMinimumWidth( w );
            d_data->buttonUp[i]->setMinimumWidth( w );
        }
    }

    return QWidget::event( event );
}

/*!
  Handle key events

  - Ctrl + Qt::Key_Home\n
    Step to minValue()
  - Ctrl + Qt::Key_End\n
    Step to maxValue()
  - Qt::Key_Up\n
    Increment by incSteps(QwtCounter::Button1)
  - Qt::Key_Down\n
    Decrement by incSteps(QwtCounter::Button1)
  - Qt::Key_PageUp\n
    Increment by incSteps(QwtCounter::Button2)
  - Qt::Key_PageDown\n
    Decrement by incSteps(QwtCounter::Button2)
  - Shift + Qt::Key_PageUp\n
    Increment by incSteps(QwtCounter::Button3)
  - Shift + Qt::Key_PageDown\n
    Decrement by incSteps(QwtCounter::Button3)

  \param event Key event
*/
void QwtCounter::keyPressEvent ( QKeyEvent *event )
{
    bool accepted = true;

    switch ( event->key() )
    {
        case Qt::Key_Home:
        {
            if ( event->modifiers() & Qt::ControlModifier )
                setValue( minValue() );
            else
                accepted = false;
            break;
        }
        case Qt::Key_End:
        {
            if ( event->modifiers() & Qt::ControlModifier )
                setValue( maxValue() );
            else
                accepted = false;
            break;
        }
        case Qt::Key_Up:
        {
            incValue( d_data->increment[0] );
            break;
        }
        case Qt::Key_Down:
        {
            incValue( -d_data->increment[0] );
            break;
        }
        case Qt::Key_PageUp:
        case Qt::Key_PageDown:
        {
            int increment = d_data->increment[0];
            if ( d_data->numButtons >= 2 )
                increment = d_data->increment[1];
            if ( d_data->numButtons >= 3 )
            {
                if ( event->modifiers() & Qt::ShiftModifier )
                    increment = d_data->increment[2];
            }
            if ( event->key() == Qt::Key_PageDown )
                increment = -increment;
            incValue( increment );
            break;
        }
        default:
        {
            accepted = false;
        }
    }

    if ( accepted )
    {
        event->accept();
        return;
    }

    QWidget::keyPressEvent ( event );
}

/*!
  Handle wheel events
  \param event Wheel event
*/
void QwtCounter::wheelEvent( QWheelEvent *event )
{
    event->accept();

    if ( d_data->numButtons <= 0 )
        return;

    int increment = d_data->increment[0];
    if ( d_data->numButtons >= 2 )
    {
        if ( event->modifiers() & Qt::ControlModifier )
            increment = d_data->increment[1];
    }
    if ( d_data->numButtons >= 3 )
    {
        if ( event->modifiers() & Qt::ShiftModifier )
            increment = d_data->increment[2];
    }

    for ( int i = 0; i < d_data->numButtons; i++ )
    {
        if ( d_data->buttonDown[i]->geometry().contains( event->pos() ) ||
            d_data->buttonUp[i]->geometry().contains( event->pos() ) )
        {
            increment = d_data->increment[i];
        }
    }

    const int wheel_delta = 120;

    int delta = event->delta();
    if ( delta >= 2 * wheel_delta )
        delta /= 2; // Never saw an abs(delta) < 240

    incValue( delta / wheel_delta * increment );
}

/*!
  Specify the number of steps by which the value
  is incremented or decremented when a specified button
  is pushed.

  \param button Button index
  \param nSteps Number of steps

  \sa incSteps()
*/
void QwtCounter::setIncSteps( QwtCounter::Button button, int nSteps )
{
    if ( button >= 0 && button < ButtonCnt )
        d_data->increment[button] = nSteps;
}

/*!
  \return the number of steps by which a specified button increments the value
  or 0 if the button is invalid.
  \param button Button index

  \sa setIncSteps()
*/
int QwtCounter::incSteps( QwtCounter::Button button ) const
{
    if ( button >= 0 && button < ButtonCnt )
        return d_data->increment[button];

    return 0;
}

/*!
  \brief Set a new value

  Calls QwtDoubleRange::setValue and does all visual updates.

  \param value New value
  \sa QwtDoubleRange::setValue()
*/

void QwtCounter::setValue( double value )
{
    QwtDoubleRange::setValue( value );

    showNum( this->value() );
    updateButtons();
}

/*!
  \brief Notify a change of value
*/
void QwtCounter::valueChange()
{
    if ( isValid() )
        showNum( value() );
    else
        d_data->valueEdit->setText( QString::null );

    updateButtons();

    if ( isValid() )
        Q_EMIT valueChanged( value() );
}

/*!
  \brief Update buttons according to the current value

  When the QwtCounter under- or over-flows, the focus is set to the smallest
  up- or down-button and counting is disabled.

  Counting is re-enabled on a button release event (mouse or space bar).
*/
void QwtCounter::updateButtons()
{
    if ( isValid() )
    {
        // 1. save enabled state of the smallest down- and up-button
        // 2. change enabled state on under- or over-flow

        for ( int i = 0; i < QwtCounter::ButtonCnt; i++ )
        {
            d_data->buttonDown[i]->setEnabled( value() > minValue() );
            d_data->buttonUp[i]->setEnabled( value() < maxValue() );
        }
    }
    else
    {
        for ( int i = 0; i < QwtCounter::ButtonCnt; i++ )
        {
            d_data->buttonDown[i]->setEnabled( false );
            d_data->buttonUp[i]->setEnabled( false );
        }
    }
}

/*!
  \brief Specify the number of buttons on each side of the label
  \param numButtons Number of buttons
*/
void QwtCounter::setNumButtons( int numButtons )
{
    if ( numButtons < 0 || numButtons > QwtCounter::ButtonCnt )
        return;

    for ( int i = 0; i < QwtCounter::ButtonCnt; i++ )
    {
        if ( i < numButtons )
        {
            d_data->buttonDown[i]->show();
            d_data->buttonUp[i]->show();
        }
        else
        {
            d_data->buttonDown[i]->hide();
            d_data->buttonUp[i]->hide();
        }
    }

    d_data->numButtons = numButtons;
}

/*!
    \return The number of buttons on each side of the widget.
*/
int QwtCounter::numButtons() const
{
    return d_data->numButtons;
}

/*!
  Display number string

  \param number Number
*/
void QwtCounter::showNum( double number )
{
    QString text;
    text.setNum( number );

    const int cursorPos = d_data->valueEdit->cursorPosition();
    d_data->valueEdit->setText( text );
    d_data->valueEdit->setCursorPosition( cursorPos );
}

//!  Button clicked
void QwtCounter::btnClicked()
{
    for ( int i = 0; i < ButtonCnt; i++ )
    {
        if ( d_data->buttonUp[i] == sender() )
            incValue( d_data->increment[i] );

        if ( d_data->buttonDown[i] == sender() )
            incValue( -d_data->increment[i] );
    }
}

//!  Button released
void QwtCounter::btnReleased()
{
    Q_EMIT buttonReleased( value() );
}

/*!
  \brief Notify change of range

  This function updates the enabled property of
  all buttons contained in QwtCounter.
*/
void QwtCounter::rangeChange()
{
    updateButtons();
}

//! A size hint
QSize QwtCounter::sizeHint() const
{
    QString tmp;

    int w = tmp.setNum( minValue() ).length();
    int w1 = tmp.setNum( maxValue() ).length();
    if ( w1 > w )
        w = w1;
    w1 = tmp.setNum( minValue() + step() ).length();
    if ( w1 > w )
        w = w1;
    w1 = tmp.setNum( maxValue() - step() ).length();
    if ( w1 > w )
        w = w1;

    tmp.fill( '9', w );

    QFontMetrics fm( d_data->valueEdit->font() );
    w = fm.width( tmp ) + 2;
    if ( d_data->valueEdit->hasFrame() )
        w += 2 * style()->pixelMetric( QStyle::PM_DefaultFrameWidth );

    // Now we replace default sizeHint contribution of d_data->valueEdit by
    // what we really need.

    w += QWidget::sizeHint().width() - d_data->valueEdit->sizeHint().width();

    const int h = qMin( QWidget::sizeHint().height(),
        d_data->valueEdit->minimumSizeHint().height() );
    return QSize( w, h );
}

//! returns the step size
double QwtCounter::step() const
{
    return QwtDoubleRange::step();
}

/*!
   Set the step size
   \param stepSize Step size
   \sa QwtDoubleRange::setStep()
*/
void QwtCounter::setStep( double stepSize )
{
    QwtDoubleRange::setStep( stepSize );
}

//! returns the minimum value of the range
double QwtCounter::minValue() const
{
    return QwtDoubleRange::minValue();
}

/*!
  Set the minimum value of the range

  \param value Minimum value
  \sa setMaxValue(), minValue()
*/
void QwtCounter::setMinValue( double value )
{
    setRange( value, maxValue(), step() );
}

//! returns the maximum value of the range
double QwtCounter::maxValue() const
{
    return QwtDoubleRange::maxValue();
}

/*!
  Set the maximum value of the range

  \param value Maximum value
  \sa setMinValue(), maxVal()
*/
void QwtCounter::setMaxValue( double value )
{
    setRange( minValue(), value, step() );
}

/*!
  Set the number of increment steps for button 1
  \param nSteps Number of steps
*/
void QwtCounter::setStepButton1( int nSteps )
{
    setIncSteps( Button1, nSteps );
}

//! returns the number of increment steps for button 1
int QwtCounter::stepButton1() const
{
    return incSteps( Button1 );
}

/*!
  Set the number of increment steps for button 2
  \param nSteps Number of steps
*/
void QwtCounter::setStepButton2( int nSteps )
{
    setIncSteps( Button2, nSteps );
}

//! returns the number of increment steps for button 2
int QwtCounter::stepButton2() const
{
    return incSteps( Button2 );
}

/*!
  Set the number of increment steps for button 3
  \param nSteps Number of steps
*/
void QwtCounter::setStepButton3( int nSteps )
{
    setIncSteps( Button3, nSteps );
}

//! returns the number of increment steps for button 3
int QwtCounter::stepButton3() const
{
    return incSteps( Button3 );
}

//! \return Current value
double QwtCounter::value() const
{
    return QwtDoubleRange::value();
}

