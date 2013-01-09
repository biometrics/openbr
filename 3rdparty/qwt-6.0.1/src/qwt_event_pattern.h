/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#ifndef QWT_EVENT_PATTERN
#define QWT_EVENT_PATTERN 1

#include "qwt_global.h"
#include <qnamespace.h>
#include <qvector.h>

class QMouseEvent;
class QKeyEvent;

/*!
  \brief A collection of event patterns

  QwtEventPattern introduces an level of indirection for mouse and
  keyboard inputs. Those are represented by symbolic names, so
  the application code can be configured by individual mappings.

  \sa QwtPicker, QwtPickerMachine, QwtPlotZoomer
*/
class QWT_EXPORT QwtEventPattern
{
public:
    /*!
      \brief Symbolic mouse input codes

      The default initialization for 3 button mice is:
      - MouseSelect1\n
        Qt::LeftButton
      - MouseSelect2\n
        Qt::RightButton
      - MouseSelect3\n
        Qt::MidButton
      - MouseSelect4\n
        Qt::LeftButton + Qt::ShiftButton
      - MouseSelect5\n
        Qt::RightButton + Qt::ShiftButton
      - MouseSelect6\n
        Qt::MidButton + Qt::ShiftButton

      The default initialization for 2 button mice is:
      - MouseSelect1\n
        Qt::LeftButton
      - MouseSelect2\n
        Qt::RightButton
      - MouseSelect3\n
        Qt::LeftButton + Qt::AltButton
      - MouseSelect4\n
        Qt::LeftButton + Qt::ShiftButton
      - MouseSelect5\n
        Qt::RightButton + Qt::ShiftButton
      - MouseSelect6\n
        Qt::LeftButton + Qt::AltButton + Qt::ShiftButton

      The default initialization for 1 button mice is:
      - MouseSelect1\n
        Qt::LeftButton
      - MouseSelect2\n
        Qt::LeftButton + Qt::ControlButton
      - MouseSelect3\n
        Qt::LeftButton + Qt::AltButton
      - MouseSelect4\n
        Qt::LeftButton + Qt::ShiftButton
      - MouseSelect5\n
        Qt::LeftButton + Qt::ControlButton + Qt::ShiftButton
      - MouseSelect6\n
        Qt::LeftButton + Qt::AltButton + Qt::ShiftButton

      \sa initMousePattern()
    */

    enum MousePatternCode
    {
        MouseSelect1,
        MouseSelect2,
        MouseSelect3,
        MouseSelect4,
        MouseSelect5,
        MouseSelect6,

        MousePatternCount
    };

    /*!
      \brief Symbolic keyboard input codes

      Default initialization:
      - KeySelect1\n
        Qt::Key_Return
      - KeySelect2\n
        Qt::Key_Space
      - KeyAbort\n
        Qt::Key_Escape

      - KeyLeft\n
        Qt::Key_Left
      - KeyRight\n
        Qt::Key_Right
      - KeyUp\n
        Qt::Key_Up
      - KeyDown\n
        Qt::Key_Down

      - KeyUndo\n
        Qt::Key_Minus
      - KeyRedo\n
        Qt::Key_Plus
      - KeyHome\n
        Qt::Key_Escape
    */
    enum KeyPatternCode
    {
        KeySelect1,
        KeySelect2,
        KeyAbort,

        KeyLeft,
        KeyRight,
        KeyUp,
        KeyDown,

        KeyRedo,
        KeyUndo,
        KeyHome,

        KeyPatternCount
    };

    //! A pattern for mouse events
    class MousePattern
    {
    public:
        //! Constructor
        MousePattern( int btn = Qt::NoButton, int st = Qt::NoButton )
        {
            button = btn;
            state = st;
        }

        //! Button code
        int button;

        //! State
        int state;
    };

    //! A pattern for key events
    class KeyPattern
    {
    public:
        //! Constructor
        KeyPattern( int k = 0, int st = Qt::NoButton )
        {
            key = k;
            state = st;
        }

        //! Key code
        int key;

        //! State
        int state;
    };

    QwtEventPattern();
    virtual ~QwtEventPattern();

    void initMousePattern( int numButtons );
    void initKeyPattern();

    void setMousePattern( uint pattern, int button, int state = Qt::NoButton );
    void setKeyPattern( uint pattern, int key, int state = Qt::NoButton );

    void setMousePattern( const QVector<MousePattern> & );
    void setKeyPattern( const QVector<KeyPattern> & );

    const QVector<MousePattern> &mousePattern() const;
    const QVector<KeyPattern> &keyPattern() const;

    QVector<MousePattern> &mousePattern();
    QVector<KeyPattern> &keyPattern();

    bool mouseMatch( uint pattern, const QMouseEvent * ) const;
    bool keyMatch( uint pattern, const QKeyEvent * ) const;

protected:
    virtual bool mouseMatch( const MousePattern &, const QMouseEvent * ) const;
    virtual bool keyMatch( const KeyPattern &, const QKeyEvent * ) const;

private:

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif
    QVector<MousePattern> d_mousePattern;
    QVector<KeyPattern> d_keyPattern;
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
};

//! Compare operator
inline bool operator==( QwtEventPattern::MousePattern b1,
    QwtEventPattern::MousePattern  b2 )
{
    return b1.button == b2.button && b1.state == b2.state;
}

//! Compare operator
inline bool operator==( QwtEventPattern::KeyPattern b1,
   QwtEventPattern::KeyPattern  b2 )
{
    return b1.key == b2.key && b1.state == b2.state;
}

#endif
