find_path(NEC3_DIR Include/NeoFacePro.h ${CMAKE_SOURCE_DIR}/3rdparty/*)

include_directories(${NEC3_DIR}/Include)
link_directories(${NEC3_DIR}/Lib)

set(NEC3_LIBS NeoFacePro)
