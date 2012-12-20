find_path(LIBMR MetaRecognition.h ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(LIBMR)
include_directories(${LIBMR})
set(LIBMR_SRC ${LIBMR}/MetaRecognition.cpp ${LIBMR}/weibull.c)
