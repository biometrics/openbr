################################################################
# Qwt Widget Library
# Copyright (C) 1997   Josef Wilgen
# Copyright (C) 2002   Uwe Rathmann
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the Qwt License, Version 1.0
################################################################

QWT_ROOT = $${PWD}/..

include( $${QWT_ROOT}/qwtconfig.pri )
include( $${QWT_ROOT}/qwtbuild.pri )

TEMPLATE  = lib

DESTDIR         = $${QWT_ROOT}/lib
INCLUDEPATH    += $${QWT_ROOT}/src
DEPENDPATH     += $${QWT_ROOT}/src

contains(QWT_CONFIG, QwtDll) {
    CONFIG += dll
    win32|symbian: DEFINES += QT_DLL QWT_DLL QWT_MAKEDLL
}   
else {
    CONFIG += staticlib
}

contains(QWT_CONFIG, QwtFramework) {

    CONFIG += lib_bundle
    LIBS      += -F$${QWT_ROOT}/lib
}
else {

    LIBS      += -L$${QWT_ROOT}/lib
}

IPATH       = $${INCLUDEPATH}
qtAddLibrary(qwt)
INCLUDEPATH = $${IPATH}

# Install directives

target.path    = $${QWT_INSTALL_LIBS}
doc.path       = $${QWT_INSTALL_DOCS}

INSTALLS       = target 
