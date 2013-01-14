################################################################
# Qwt Widget Library
# Copyright (C) 1997   Josef Wilgen
# Copyright (C) 2002   Uwe Rathmann
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the Qwt License, Version 1.0
################################################################

include( $${PWD}/../qwtconfig.pri )

TEMPLATE = subdirs

contains(QWT_CONFIG, QwtPlot) {
    
    SUBDIRS += \
        cpuplot \
        curvdemo1   \
        curvdemo2 \
        friedberg \
        sinusplot \
        realtime \
        refreshtest \
        navigation \
        plotmatrix \
        spectrogram \
        rasterview \
        tvplot 

    contains(QWT_CONFIG, QwtWidgets) {

        SUBDIRS += \
            bode \
            event_filter \
            oscilloscope  
    }
    
    contains(QWT_CONFIG, QwtSvg) {

        SUBDIRS += \
            svgmap
    }
}

contains(QWT_CONFIG, QwtWidgets) {

    SUBDIRS += \
        sysinfo \
        radio \
        dials \
        sliders
}
