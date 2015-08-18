set(BR_WITH_QTNETWORK ON CACHE BOOL "Build with QtNetwork")
add_definitions(-DBR_WITH_QTNETWORK)

if (${BR_EMBEDDED})
  set(BR_WITH_QTNETWORK OFF)
endif()

if(${BR_WITH_QTNETWORK})
  find_package(Qt5Network)
  find_package(HttpParser)
  set(QT_DEPENDENCIES ${QT_DEPENDENCIES} Network)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${HTTPPARSER_SRC})
  install(FILES ${HTTPPARSER_LICENSE} RENAME http-parser DESTINATION share/openbr/licenses)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/format/url.cpp)
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/format/post.cpp)
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/gallery/post.cpp)
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/gallery/google.cpp)
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/io/download.cpp)
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/core/processwrapper.cpp)
endif()
