option(BR_WITH_QTNETWORK "Build with QtNetwork" ON)
if(${BR_WITH_QTNETWORK})
  find_package(Qt5Network)
  find_package(HttpParser)
  set(QT_DEPENDENCIES ${QT_DEPENDENCIES} Network)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${HTTPPARSER_SRC})
  if(${BR_INSTALL_SHARE})
    install(FILES ${HTTPPARSER_LICENSE} RENAME http-parser DESTINATION share/openbr/licenses)
  endif()
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/core/processwrapper.cpp
                                                 plugins/io/download.cpp
                                                 plugins/format/url.cpp
                                                 plugins/format/post.cpp
                                                 plugins/gallery/google.cpp
                                                 plugins/gallery/post.cpp)
endif()
