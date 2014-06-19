set(QHTTPSERVER_DIR "${CMAKE_SOURCE_DIR}/3rdparty/qhttpserver")
set(QHTTPSERVER_SRC ${QHTTPSERVER_DIR}/src/qhttpconnection.cpp
                    ${QHTTPSERVER_DIR}/src/qhttpresponse.cpp
                    ${QHTTPSERVER_DIR}/src/qhttprequest.cpp
                    ${QHTTPSERVER_DIR}/src/qhttpserver.cpp
                    ${QHTTPSERVER_DIR}/http-parser/http_parser.c)
add_definitions(-DQHTTPSERVER_EXPORT)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/qhttpserver/src
                    ${CMAKE_SOURCE_DIR}/3rdparty/qhttpserver/http-parser
                    ${CMAKE_CURRENT_BINARY_DIR})
add_executable(br-serve br-serve.cpp ${QHTTPSERVER_SRC} ${BR_RESOURCES})
target_link_libraries(br-serve openbr ${BR_THIRDPARTY_LIBS})
qt5_use_modules(br-serve ${QT_DEPENDENCIES})
install(TARGETS br-serve RUNTIME DESTINATION bin)
