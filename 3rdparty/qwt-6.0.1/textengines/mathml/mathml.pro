################################################################
# Qwt Widget Library
# Copyright (C) 1997   Josef Wilgen
# Copyright (C) 2002   Uwe Rathmann
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the Qwt License, Version 1.0
################################################################

message(The qwtmathml library contains code of the MML Widget from the Qt solutions package.)
message(Beside the Qwt license you also have to take care of its license. )

include( $${PWD}/../textengines.pri )

TARGET    = $$qtLibraryTarget(qwtmathml)
QT       += xml

HEADERS = \
    qwt_mathml_text_engine.h

SOURCES = \
    qwt_mathml_text_engine.cpp

# qwt_mml_document.h/qwt_mml_document.cpp has been stripped down from
# the mathml widgets offered in the Qt solutions package. 

HEADERS += qwt_mml_document.h
SOURCES += qwt_mml_document.cpp

qwtmathmlspec.files  = qwtmathml.prf
qwtmathmlspec.path  = $${QWT_INSTALL_FEATURES}

INSTALLS += qwtmathmlspec

CONFIG(lib_bundle) {

    FRAMEWORK_HEADERS.version = Versions
    FRAMEWORK_HEADERS.files   = qwt_mathml_text_engine.h
    FRAMEWORK_HEADERS.path    = Headers
    QMAKE_BUNDLE_DATA        += FRAMEWORK_HEADERS
}
else {

    headers.files  = qwt_mathml_text_engine.h
    headers.path   = $${QWT_INSTALL_HEADERS}
    INSTALLS       += headers
}

