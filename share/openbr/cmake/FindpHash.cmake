find_path(PHASH_DIR src/pHash.h ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(PHASH_DIR)
include_directories(${PHASH_DIR} ${PHASH_DIR}/src)
set(PHASH_SRC ${PHASH_DIR}/src/pHash.cpp ${PHASH_DIR}/src/ph_fft.c)
