################################################################
# Qwt Widget Library
# Copyright (C) 1997   Josef Wilgen
# Copyright (C) 2002   Uwe Rathmann
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the Qwt License, Version 1.0
################################################################

# Copied and modified from qt_functions.prf

defineReplace(qwtLibraryTarget) {

   unset(LIBRARY_NAME)
   LIBRARY_NAME = $$1

   mac:!static:contains(QT_CONFIG, qt_framework) {
      QMAKE_FRAMEWORK_BUNDLE_NAME = $$LIBRARY_NAME
      export(QMAKE_FRAMEWORK_BUNDLE_NAME)
   }

   contains(TEMPLATE, .*lib):CONFIG(debug, debug|release) {
      !debug_and_release|build_pass {
          mac:RET = $$member(LIBRARY_NAME, 0)_debug
          else:win32:RET = $$member(LIBRARY_NAME, 0)d
      }
   }

   isEmpty(RET):RET = $$LIBRARY_NAME
   return($$RET)
}

defineTest(qwtAddLibrary) {

    LIB_NAME = $$1

    unset(LINKAGE)

    if(!debug_and_release|build_pass):CONFIG(debug, debug|release) {
       win32:LINKAGE = -l$${LIB_NAME}$${QT_LIBINFIX}d
       mac:LINKAGE = -l$${LIB_NAME}$${QT_LIBINFIX}_debug
    }

    isEmpty(LINKAGE):LINKAGE = -l$${LIB_NAME}$${QT_LIBINFIX}

    !isEmpty(QMAKE_LSB) {
        QMAKE_LFLAGS *= --lsb-shared-libs=$${LIB_NAME}$${QT_LIBINFIX}
    }

    LIBS += $$LINKAGE
    export(LIBS)
    export(QMAKE_LFLAGS)

    return(true)
}
