find_path(NEC3_DIR Include/NeoFacePro.h ${CMAKE_SOURCE_DIR}/3rdparty/*)

include_directories(${NEC3_DIR}/Include)
link_directories(${NEC3_DIR}/Lib)

#if(MSVC)
#  file(GLOB NEC3_LIBS ${NECLATENT1_LIB_DIR}/*.lib)
#else()
#  file(GLOB NEC3_LIBS ${NECLATENT1_BIN_DIR}/*.dll)
#endif()

set(NEC3_LIBS NeoFacePro)
