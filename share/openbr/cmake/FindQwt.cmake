find_path(QWT_DIR qwt.pro ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(QWT_DIR)
include_directories(${QWT_DIR}/src)
aux_source_directory(${QWT_DIR}/src QWT_SRC)
