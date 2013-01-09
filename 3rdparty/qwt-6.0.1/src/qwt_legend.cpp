/* -*- mode: C++ ; c-file-style: "stroustrup" -*- *****************************
 * Qwt Widget Library
 * Copyright (C) 1997   Josef Wilgen
 * Copyright (C) 2002   Uwe Rathmann
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the Qwt License, Version 1.0
 *****************************************************************************/

#include "qwt_legend.h"
#include "qwt_legend_itemmanager.h"
#include "qwt_legend_item.h"
#include "qwt_dyngrid_layout.h"
#include "qwt_math.h"
#include <qapplication.h>
#include <qmap.h>
#include <qscrollbar.h>
#include <qscrollarea.h>

class QwtLegend::PrivateData
{
public:
    class LegendMap
    {
    public:
        void insert( const QwtLegendItemManager *, QWidget * );

        void remove( const QwtLegendItemManager * );
        void remove( QWidget * );

        void clear();

        uint count() const;

        inline const QWidget *find( const QwtLegendItemManager * ) const;
        inline QWidget *find( const QwtLegendItemManager * );

        inline const QwtLegendItemManager *find( const QWidget * ) const;
        inline QwtLegendItemManager *find( const QWidget * );

        const QMap<QWidget *, const QwtLegendItemManager *> &widgetMap() const;
        QMap<QWidget *, const QwtLegendItemManager *> &widgetMap();

    private:
        QMap<QWidget *, const QwtLegendItemManager *> d_widgetMap;
        QMap<const QwtLegendItemManager *, QWidget *> d_itemMap;
    };

    QwtLegend::LegendItemMode itemMode;

    LegendMap map;

    class LegendView;
    LegendView *view;
};

class QwtLegend::PrivateData::LegendView: public QScrollArea
{
public:
    LegendView( QWidget *parent ):
        QScrollArea( parent )
    {
        setFocusPolicy( Qt::NoFocus );

        contentsWidget = new QWidget( this );
        contentsWidget->setObjectName( "QwtLegendViewContents" );

        setWidget( contentsWidget );
        setWidgetResizable( false );

        viewport()->setObjectName( "QwtLegendViewport" );

        // QScrollArea::setWidget internally sets autoFillBackground to true
        // But we don't want a background.
        contentsWidget->setAutoFillBackground( false );
        viewport()->setAutoFillBackground( false );
    }

    virtual bool viewportEvent( QEvent *e )
    {
        bool ok = QScrollArea::viewportEvent( e );

        if ( e->type() == QEvent::Resize )
        {
            QEvent event( QEvent::LayoutRequest );
            QApplication::sendEvent( contentsWidget, &event );
        }
        return ok;
    }

    QSize viewportSize( int w, int h ) const
    {
        const int sbHeight = horizontalScrollBar()->sizeHint().height();
        const int sbWidth = verticalScrollBar()->sizeHint().width();

        const int cw = contentsRect().width();
        const int ch = contentsRect().height();

        int vw = cw;
        int vh = ch;

        if ( w > vw )
            vh -= sbHeight;

        if ( h > vh )
        {
            vw -= sbWidth;
            if ( w > vw && vh == ch )
                vh -= sbHeight;
        }
        return QSize( vw, vh );
    }

    QWidget *contentsWidget;
};

void QwtLegend::PrivateData::LegendMap::insert(
    const QwtLegendItemManager *item, QWidget *widget )
{
    d_itemMap.insert( item, widget );
    d_widgetMap.insert( widget, item );
}

void QwtLegend::PrivateData::LegendMap::remove( const QwtLegendItemManager *item )
{
    QWidget *widget = d_itemMap[item];
    d_itemMap.remove( item );
    d_widgetMap.remove( widget );
}

void QwtLegend::PrivateData::LegendMap::remove( QWidget *widget )
{
    const QwtLegendItemManager *item = d_widgetMap[widget];
    d_itemMap.remove( item );
    d_widgetMap.remove( widget );
}

void QwtLegend::PrivateData::LegendMap::clear()
{

    /*
       We can't delete the widgets in the following loop, because
       we would get ChildRemoved events, changing d_itemMap, while
       we are iterating.
     */

    QList<const QWidget *> widgets;

    QMap<const QwtLegendItemManager *, QWidget *>::const_iterator it;
    for ( it = d_itemMap.begin(); it != d_itemMap.end(); ++it )
        widgets.append( it.value() );

    d_itemMap.clear();
    d_widgetMap.clear();

    for ( int i = 0; i < widgets.size(); i++ )
        delete widgets[i];
}

uint QwtLegend::PrivateData::LegendMap::count() const
{
    return d_itemMap.count();
}

inline const QWidget *QwtLegend::PrivateData::LegendMap::find( 
    const QwtLegendItemManager *item ) const
{
    if ( !d_itemMap.contains( item ) )
        return NULL;

    return d_itemMap[item];
}

inline QWidget *QwtLegend::PrivateData::LegendMap::find( 
    const QwtLegendItemManager *item )
{
    if ( !d_itemMap.contains( item ) )
        return NULL;

    return d_itemMap[item];
}

inline const QwtLegendItemManager *QwtLegend::PrivateData::LegendMap::find(
    const QWidget *widget ) const
{
    QWidget *w = const_cast<QWidget *>( widget );
    if ( !d_widgetMap.contains( w ) )
        return NULL;

    return d_widgetMap[w];
}

inline QwtLegendItemManager *QwtLegend::PrivateData::LegendMap::find(
    const QWidget *widget )
{
    QWidget *w = const_cast<QWidget *>( widget );
    if ( !d_widgetMap.contains( w ) )
        return NULL;

    return const_cast<QwtLegendItemManager *>( d_widgetMap[w] );
}

inline const QMap<QWidget *, const QwtLegendItemManager *> &
QwtLegend::PrivateData::LegendMap::widgetMap() const
{
    return d_widgetMap;
}

inline QMap<QWidget *, const QwtLegendItemManager *> &
QwtLegend::PrivateData::LegendMap::widgetMap()
{
    return d_widgetMap;
}

/*!
  Constructor

  \param parent Parent widget
*/
QwtLegend::QwtLegend( QWidget *parent ):
    QFrame( parent )
{
    setFrameStyle( NoFrame );

    d_data = new QwtLegend::PrivateData;
    d_data->itemMode = QwtLegend::ReadOnlyItem;

    d_data->view = new QwtLegend::PrivateData::LegendView( this );
    d_data->view->setObjectName( "QwtLegendView" );
    d_data->view->setFrameStyle( NoFrame );

    QwtDynGridLayout *gridLayout = new QwtDynGridLayout(
        d_data->view->contentsWidget );
    gridLayout->setAlignment( Qt::AlignHCenter | Qt::AlignTop );

    d_data->view->contentsWidget->installEventFilter( this );

    QVBoxLayout *layout = new QVBoxLayout( this );
    layout->setContentsMargins( 0, 0, 0, 0 );
    layout->addWidget( d_data->view );
}

//! Destructor
QwtLegend::~QwtLegend()
{
    delete d_data;
}

//! \sa LegendItemMode
void QwtLegend::setItemMode( LegendItemMode mode )
{
    d_data->itemMode = mode;
}

//! \sa LegendItemMode
QwtLegend::LegendItemMode QwtLegend::itemMode() const
{
    return d_data->itemMode;
}

/*!
  The contents widget is the only child of the viewport of 
  the internal QScrollArea  and the parent widget of all legend items.

  \return Container widget of the legend items
*/
QWidget *QwtLegend::contentsWidget()
{
    return d_data->view->contentsWidget;
}

/*!
  \return Horizontal scrollbar
  \sa verticalScrollBar()
*/
QScrollBar *QwtLegend::horizontalScrollBar() const
{
    return d_data->view->horizontalScrollBar();
}

/*!
  \return Vertical scrollbar
  \sa horizontalScrollBar()
*/
QScrollBar *QwtLegend::verticalScrollBar() const
{
    return d_data->view->verticalScrollBar();
}

/*!
  The contents widget is the only child of the viewport of 
  the internal QScrollArea  and the parent widget of all legend items.

  \return Container widget of the legend items

*/
const QWidget *QwtLegend::contentsWidget() const
{
    return d_data->view->contentsWidget;
}

/*!
  Insert a new item for a plot item
  \param plotItem Plot item
  \param legendItem New legend item
  \note The parent of item will be changed to contentsWidget()
*/
void QwtLegend::insert( const QwtLegendItemManager *plotItem, QWidget *legendItem )
{
    if ( legendItem == NULL || plotItem == NULL )
        return;

    QWidget *contentsWidget = d_data->view->contentsWidget;

    if ( legendItem->parent() != contentsWidget )
        legendItem->setParent( contentsWidget );

    legendItem->show();

    d_data->map.insert( plotItem, legendItem );

    layoutContents();

    if ( contentsWidget->layout() )
    {
        contentsWidget->layout()->addWidget( legendItem );

        // set tab focus chain

        QWidget *w = NULL;

        for ( int i = 0; i < contentsWidget->layout()->count(); i++ )
        {
            QLayoutItem *item = contentsWidget->layout()->itemAt( i );
            if ( w && item->widget() )
                QWidget::setTabOrder( w, item->widget() );

            w = item->widget();
        }
    }
    if ( parentWidget() && parentWidget()->layout() == NULL )
    {
        /*
           updateGeometry() doesn't post LayoutRequest in certain
           situations, like when we are hidden. But we want the
           parent widget notified, so it can show/hide the legend
           depending on its items.
         */
        QApplication::postEvent( parentWidget(),
            new QEvent( QEvent::LayoutRequest ) );
    }
}

/*!
  Find the widget that represents a plot item

  \param plotItem Plot item
  \return Widget on the legend, or NULL
*/
QWidget *QwtLegend::find( const QwtLegendItemManager *plotItem ) const
{
    return d_data->map.find( plotItem );
}

/*!
  Find the widget that represents a plot item

  \param legendItem Legend item
  \return Widget on the legend, or NULL
*/
QwtLegendItemManager *QwtLegend::find( const QWidget *legendItem ) const
{
    return d_data->map.find( legendItem );
}

/*!
   Find the corresponding item for a plotItem and remove it
   from the item list.

   \param plotItem Plot item
*/
void QwtLegend::remove( const QwtLegendItemManager *plotItem )
{
    QWidget *legendItem = d_data->map.find( plotItem );
    d_data->map.remove( legendItem );
    delete legendItem;
}

//! Remove all items.
void QwtLegend::clear()
{
    bool doUpdate = updatesEnabled();
    if ( doUpdate )
        setUpdatesEnabled( false );

    d_data->map.clear();

    if ( doUpdate )
        setUpdatesEnabled( true );

    update();
}

//! Return a size hint.
QSize QwtLegend::sizeHint() const
{
    QSize hint = d_data->view->contentsWidget->sizeHint();
    hint += QSize( 2 * frameWidth(), 2 * frameWidth() );

    return hint;
}

/*!
  \return The preferred height, for the width w.
  \param width Width
*/
int QwtLegend::heightForWidth( int width ) const
{
    width -= 2 * frameWidth();

    int h = d_data->view->contentsWidget->heightForWidth( width );
    if ( h >= 0 )
        h += 2 * frameWidth();

    return h;
}

/*!
  Adjust contents widget and item layout to the size of the viewport().
*/
void QwtLegend::layoutContents()
{
    const QSize visibleSize = 
        d_data->view->viewport()->contentsRect().size();

    const QwtDynGridLayout *tl = qobject_cast<QwtDynGridLayout *>(
        d_data->view->contentsWidget->layout() );
    if ( tl )
    {
        const int minW = int( tl->maxItemWidth() ) + 2 * tl->margin();

        int w = qMax( visibleSize.width(), minW );
        int h = qMax( tl->heightForWidth( w ), visibleSize.height() );

        const int vpWidth = d_data->view->viewportSize( w, h ).width();
        if ( w > vpWidth )
        {
            w = qMax( vpWidth, minW );
            h = qMax( tl->heightForWidth( w ), visibleSize.height() );
        }

        d_data->view->contentsWidget->resize( w, h );
    }
}

/*!
  Handle QEvent::ChildRemoved andQEvent::LayoutRequest events 
  for the contentsWidget().

  \param object Object to be filtered
  \param event Event
*/
bool QwtLegend::eventFilter( QObject *object, QEvent *event )
{
    if ( object == d_data->view->contentsWidget )
    {
        switch ( event->type() )
        {
            case QEvent::ChildRemoved:
            {
                const QChildEvent *ce = 
                    static_cast<const QChildEvent *>(event);
                if ( ce->child()->isWidgetType() )
                {
                    QWidget *w = static_cast< QWidget * >( ce->child() );
                    d_data->map.remove( w );
                }
                break;
            }
            case QEvent::LayoutRequest:
            {
                layoutContents();
                break;
            }
            default:
                break;
        }
    }

    return QFrame::eventFilter( object, event );
}


//! Return true, if there are no legend items.
bool QwtLegend::isEmpty() const
{
    return d_data->map.count() == 0;
}

//! Return the number of legend items.
uint QwtLegend::itemCount() const
{
    return d_data->map.count();
}

//! Return a list of all legend items
QList<QWidget *> QwtLegend::legendItems() const
{
    const QMap<QWidget *, const QwtLegendItemManager *> &map =
        d_data->map.widgetMap();

    QList<QWidget *> list;

    QMap<QWidget *, const QwtLegendItemManager *>::const_iterator it;
    for ( it = map.begin(); it != map.end(); ++it )
        list += it.key();

    return list;
}