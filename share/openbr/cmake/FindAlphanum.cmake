find_path(ALPHANUM_DIR alphanum.hpp ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(ALPHANUM_DIR)
include_directories(${ALPHANUM_DIR})
