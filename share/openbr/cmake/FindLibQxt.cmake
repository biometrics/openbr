find_path(LIBQXT_DIR libqxt.pro ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(LIBQXT_DIR)
include_directories(${LIBQXT_DIR}/include/QxtBerkeley
                    ${LIBQXT_DIR}/include/QxtCore
                    ${LIBQXT_DIR}/include/QxtGui
                    ${LIBQXT_DIR}/include/QxtNetwork
                    ${LIBQXT_DIR}/include/QxtSql
                    ${LIBQXT_DIR}/include/QxtWeb
                    ${LIBQXT_DIR}/include/QxtZeroconf
                    ${LIBQXT_DIR}/src/berkeley
                    ${LIBQXT_DIR}/src/core
                    ${LIBQXT_DIR}/src/gui
                    ${LIBQXT_DIR}/src/network
                    ${LIBQXT_DIR}/src/sql
                    ${LIBQXT_DIR}/src/web
                    ${LIBQXT_DIR}/src/zeroconf)
set(LIBQXT_SRC ${LIBQXT}/src/core/qxtcsvmodel.cpp)
find_package(Qt4 REQUIRED)
qt4_wrap_cpp(LIBQXT_MOC ${LIBQXT}/src/core/qxtcsvmodel.h)
set(LIBQXT_SRC ${LIBQXT_SRC} ${LIBQXT_MOC})
set(LIBQXT_LICENSE ${LIBQXT_DIR}/lgpl-2.1.txt)
