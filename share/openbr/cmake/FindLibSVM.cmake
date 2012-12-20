find_path(LibSVM_DIR svm.h ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(LibSVM_DIR)
include_directories(${LibSVM_DIR})
set(LibSVM_SRC ${LibSVM_DIR}/svm.cpp)
