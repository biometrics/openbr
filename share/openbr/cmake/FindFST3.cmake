find_path(FST3_DIR _src_criteria/classifier.hpp ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(FST3_DIR)
include_directories(${FST3_DIR}/_src_criteria ${FST3_DIR}/_src_dataio ${FST3_DIR}/_src_global ${FST3_DIR}/_src_search)
set(FST3_SRC ${FST3_DIR}/_src_global/global.cpp)
