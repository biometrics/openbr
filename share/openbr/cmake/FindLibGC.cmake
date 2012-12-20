find_path(LIBGC_DIR gc.hpp ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(LIBGC_DIR)
include_directories(${LIBGC_DIR})
