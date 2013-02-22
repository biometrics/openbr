find_path(QWT_DIR qwt.pro ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(QWT_DIR)
include_directories(${QWT_DIR}/src)
file(GLOB QWT_SRC ${QWT_DIR}/src/*.cpp)
