find_path(CIMG_DIR CImg.h ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(CIMG_DIR)
include_directories(${CIMG_DIR})
