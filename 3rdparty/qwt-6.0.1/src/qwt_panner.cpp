/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_panner.h"
#include "qwt_picker.h"
#include <qpainter.h>
#include <qpixmap.h>
#include <qevent.h>
#include <qcursor.h>
#include <qbitmap.h>

static QVector<QwtPicker *> qwtActivePickers( QWidget *w )
{
    QVector<QwtPicker *> pickers;

    QObjectList children = w->children();
    for ( int i = 0; i < children.size(); i++ )
    {
        QObject *obj = children[i];
        if ( obj->inherits( "QwtPicker" ) )
        {
            QwtPicker *picker = ( QwtPicker * )obj;
            if ( picker->isEnabled() )
                pickers += picker;
        }
    }

    return pickers;
}

class QwtPanner::PrivateData
{
public:
    PrivateData():
        button( Qt::LeftButton ),
        buttonState( Qt::NoButton ),
        abortKey( Qt::Key_Escape ),
        abortKeyState( Qt::NoButton ),
#ifndef QT_NO_CURSOR
        cursor( NULL ),
        restoreCursor( NULL ),
        hasCursor( false ),
#endif
        isEnabled( false )
    {
        orientations = Qt::Vertical | Qt::Horizontal;
    }

    ~PrivateData()
    {
#ifndef QT_NO_CURSOR
        delete cursor;
        delete restoreCursor;
#endif
    }

    int button;
    int buttonState;
    int abortKey;
    int abortKeyState;

    QPoint initialPos;
    QPoint pos;

    QPixmap pixmap;
    QBitmap contentsMask;

#ifndef QT_NO_CURSOR
    QCursor *cursor;
    QCursor *restoreCursor;
    bool hasCursor;
#endif
    bool isEnabled;
    Qt::Orientations orientations;
};

/*!
  Creates an panner that is enabled for the left mouse button.

  \param parent Parent widget to be panned
*/
QwtPanner::QwtPanner( QWidget *parent ):
    QWidget( parent )
{
    d_data = new PrivateData();

    setAttribute( Qt::WA_TransparentForMouseEvents );
    setAttribute( Qt::WA_NoSystemBackground );
    setFocusPolicy( Qt::NoFocus );
    hide();

    setEnabled( true );
}

//! Destructor
QwtPanner::~QwtPanner()
{
    delete d_data;
}

/*!
   Change the mouse button
   The defaults are Qt::LeftButton and Qt::NoButton
*/
void QwtPanner::setMouseButton( int button, int buttonState )
{
    d_data->button = button;
    d_data->buttonState = buttonState;
}

//! Get the mouse button
void QwtPanner::getMouseButton( int &button, int &buttonState ) const
{
    button = d_data->button;
    buttonState = d_data->buttonState;
}

/*!
   Change the abort key
   The defaults are Qt::Key_Escape and Qt::NoButton

   \param key Key ( See Qt::Keycode )
   \param state State
*/
void QwtPanner::setAbortKey( int key, int state )
{
    d_data->abortKey = key;
    d_data->abortKeyState = state;
}

//! Get the abort key
void QwtPanner::getAbortKey( int &key, int &state ) const
{
    key = d_data->abortKey;
    state = d_data->abortKeyState;
}

/*!
   Change the cursor, that is active while panning
   The default is the cursor of the parent widget.

   \param cursor New cursor

   \sa setCursor()
*/
#ifndef QT_NO_CURSOR
void QwtPanner::setCursor( const QCursor &cursor )
{
    d_data->cursor = new QCursor( cursor );
}
#endif

/*!
   \return Cursor that is active while panning
   \sa setCursor()
*/
#ifndef QT_NO_CURSOR
const QCursor QwtPanner::cursor() const
{
    if ( d_data->cursor )
        return *d_data->cursor;

    if ( parentWidget() )
        return parentWidget()->cursor();

    return QCursor();
}
#endif

/*!
  \brief En/disable the panner

  When enabled is true an event filter is installed for
  the observed widget, otherwise the event filter is removed.

  \param on true or false
  \sa isEnabled(), eventFilter()
*/
void QwtPanner::setEnabled( bool on )
{
    if ( d_data->isEnabled != on )
    {
        d_data->isEnabled = on;

        QWidget *w = parentWidget();
        if ( w )
        {
            if ( d_data->isEnabled )
            {
                w->installEventFilter( this );
            }
            else
            {
                w->removeEventFilter( this );
                hide();
            }
        }
    }
}

/*!
   Set the orientations, where panning is enabled
   The default value is in both directions: Qt::Horizontal | Qt::Vertical

   /param o Orientation
*/
void QwtPanner::setOrientations( Qt::Orientations o )
{
    d_data->orientations = o;
}

//! Return the orientation, where paning is enabled
Qt::Orientations QwtPanner::orientations() const
{
    return d_data->orientations;
}

/*!
   Return true if a orientatio is enabled
   \sa orientations(), setOrientations()
*/
bool QwtPanner::isOrientationEnabled( Qt::Orientation o ) const
{
    return d_data->orientations & o;
}

/*!
  \return true when enabled, false otherwise
  \sa setEnabled, eventFilter()
*/
bool QwtPanner::isEnabled() const
{
    return d_data->isEnabled;
}

/*!
   \brief Paint event

   Repaint the grabbed pixmap on its current position and
   fill the empty spaces by the background of the parent widget.

   \param pe Paint event
*/
void QwtPanner::paintEvent( QPaintEvent *pe )
{
    int dx = d_data->pos.x() - d_data->initialPos.x();
    int dy = d_data->pos.y() - d_data->initialPos.y();

    QRect r( 0, 0, d_data->pixmap.width(), d_data->pixmap.height() );
    r.moveCenter( QPoint( r.center().x() + dx, r.center().y() + dy ) );

    QPixmap pm( size() );
    pm.fill( parentWidget(), 0, 0 );

    QPainter painter( &pm );

    if ( !d_data->contentsMask.isNull() )
    {
        QPixmap masked = d_data->pixmap;
        masked.setMask( d_data->contentsMask );
        painter.drawPixmap( r, masked );
    }
    else
    {
        painter.drawPixmap( r, d_data->pixmap );
    }

    painter.end();

    if ( !d_data->contentsMask.isNull() )
        pm.setMask( d_data->contentsMask );

    painter.begin( this );
    painter.setClipRegion( pe->region() );
    painter.drawPixmap( 0, 0, pm );
}

/*!
  \brief Calculate a mask for the contents of the panned widget

  Sometimes only parts of the contents of a widget should be
  panned. F.e. for a widget with a styled background with rounded borders
  only the area inside of the border should be panned.

  \return An empty bitmap, indicating no mask
*/
QBitmap QwtPanner::contentsMask() const
{
    return QBitmap();
}

/*!
  Grab the widget into a pixmap.
*/
QPixmap QwtPanner::grab() const
{
    return QPixmap::grabWidget( parentWidget() );
}

/*!
  \brief Event filter

  When isEnabled() the mouse events of the observed widget are filtered.

  \param object Object to be filtered
  \param event Event

  \sa widgetMousePressEvent(), widgetMouseReleaseEvent(),
      widgetMouseMoveEvent()
*/
bool QwtPanner::eventFilter( QObject *object, QEvent *event )
{
    if ( object == NULL || object != parentWidget() )
        return false;

    switch ( event->type() )
    {
        case QEvent::MouseButtonPress:
        {
            widgetMousePressEvent( ( QMouseEvent * )event );
            break;
        }
        case QEvent::MouseMove:
        {
            widgetMouseMoveEvent( ( QMouseEvent * )event );
            break;
        }
        case QEvent::MouseButtonRelease:
        {
            widgetMouseReleaseEvent( ( QMouseEvent * )event );
            break;
        }
        case QEvent::KeyPress:
        {
            widgetKeyPressEvent( ( QKeyEvent * )event );
            break;
        }
        case QEvent::KeyRelease:
        {
            widgetKeyReleaseEvent( ( QKeyEvent * )event );
            break;
        }
        case QEvent::Paint:
        {
            if ( isVisible() )
                return true;
            break;
        }
        default:;
    }

    return false;
}

/*!
  Handle a mouse press event for the observed widget.

  \param mouseEvent Mouse event
  \sa eventFilter(), widgetMouseReleaseEvent(),
      widgetMouseMoveEvent(),
*/
void QwtPanner::widgetMousePressEvent( QMouseEvent *mouseEvent )
{
    if ( mouseEvent->button() != d_data->button )
        return;

    QWidget *w = parentWidget();
    if ( w == NULL )
        return;

    if ( ( mouseEvent->modifiers() & Qt::KeyboardModifierMask ) !=
        ( int )( d_data->buttonState & Qt::KeyboardModifierMask ) )
    {
        return;
    }

#ifndef QT_NO_CURSOR
    showCursor( true );
#endif

    d_data->initialPos = d_data->pos = mouseEvent->pos();

    setGeometry( parentWidget()->rect() );

    // We don't want to grab the picker !
    QVector<QwtPicker *> pickers = qwtActivePickers( parentWidget() );
    for ( int i = 0; i < ( int )pickers.size(); i++ )
        pickers[i]->setEnabled( false );

    d_data->pixmap = grab();
    d_data->contentsMask = contentsMask();

    for ( int i = 0; i < ( int )pickers.size(); i++ )
        pickers[i]->setEnabled( true );

    show();
}

/*!
  Handle a mouse move event for the observed widget.

  \param mouseEvent Mouse event
  \sa eventFilter(), widgetMousePressEvent(), widgetMouseReleaseEvent()
*/
void QwtPanner::widgetMouseMoveEvent( QMouseEvent *mouseEvent )
{
    if ( !isVisible() )
        return;

    QPoint pos = mouseEvent->pos();
    if ( !isOrientationEnabled( Qt::Horizontal ) )
        pos.setX( d_data->initialPos.x() );
    if ( !isOrientationEnabled( Qt::Vertical ) )
        pos.setY( d_data->initialPos.y() );

    if ( pos != d_data->pos && rect().contains( pos ) )
    {
        d_data->pos = pos;
        update();

        Q_EMIT moved( d_data->pos.x() - d_data->initialPos.x(),
            d_data->pos.y() - d_data->initialPos.y() );
    }
}

/*!
  Handle a mouse release event for the observed widget.

  \param mouseEvent Mouse event
  \sa eventFilter(), widgetMousePressEvent(),
      widgetMouseMoveEvent(),
*/
void QwtPanner::widgetMouseReleaseEvent( QMouseEvent *mouseEvent )
{
    if ( isVisible() )
    {
        hide();
#ifndef QT_NO_CURSOR
        showCursor( false );
#endif

        QPoint pos = mouseEvent->pos();
        if ( !isOrientationEnabled( Qt::Horizontal ) )
            pos.setX( d_data->initialPos.x() );
        if ( !isOrientationEnabled( Qt::Vertical ) )
            pos.setY( d_data->initialPos.y() );

        d_data->pixmap = QPixmap();
        d_data->contentsMask = QBitmap();
        d_data->pos = pos;

        if ( d_data->pos != d_data->initialPos )
        {
            Q_EMIT panned( d_data->pos.x() - d_data->initialPos.x(),
                d_data->pos.y() - d_data->initialPos.y() );
        }
    }
}

/*!
  Handle a key press event for the observed widget.

  \param keyEvent Key event
  \sa eventFilter(), widgetKeyReleaseEvent()
*/
void QwtPanner::widgetKeyPressEvent( QKeyEvent *keyEvent )
{
    if ( keyEvent->key() == d_data->abortKey )
    {
        const bool matched =
            ( keyEvent->modifiers() & Qt::KeyboardModifierMask ) ==
                ( int )( d_data->abortKeyState & Qt::KeyboardModifierMask );
        if ( matched )
        {
            hide();
#ifndef QT_NO_CURSOR
            showCursor( false );
#endif
            d_data->pixmap = QPixmap();
        }
    }
}

/*!
  Handle a key release event for the observed widget.

  \param keyEvent Key event
  \sa eventFilter(), widgetKeyReleaseEvent()
*/
void QwtPanner::widgetKeyReleaseEvent( QKeyEvent *keyEvent )
{
    Q_UNUSED( keyEvent );
}

#ifndef QT_NO_CURSOR
void QwtPanner::showCursor( bool on )
{
    if ( on == d_data->hasCursor )
        return;

    QWidget *w = parentWidget();
    if ( w == NULL || d_data->cursor == NULL )
        return;

    d_data->hasCursor = on;

    if ( on )
    {
        if ( w->testAttribute( Qt::WA_SetCursor ) )
        {
            delete d_data->restoreCursor;
            d_data->restoreCursor = new QCursor( w->cursor() );
        }
        w->setCursor( *d_data->cursor );
    }
    else
    {
        if ( d_data->restoreCursor )
        {
            w->setCursor( *d_data->restoreCursor );
            delete d_data->restoreCursor;
            d_data->restoreCursor = NULL;
        }
        else
            w->unsetCursor();
    }
}
#endif
