find_path(DLib_DIR dlib ${PROJECT_SOURCE_DIR}/3rdparty/*)

mark_as_advanced(DLib_DIR)
include_directories(${DLib_DIR})

set(DLib_LIBS dlib)
