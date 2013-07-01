# ================================================================
# The Stasm CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(Stasm4 REQUIRED)
#   target_link_libraries(MY_TARGET ${Stasm4_LIBS})
# ================================================================

find_path(Stasm_DIR stasm/stasm_lib.h ${CMAKE_SOURCE_DIR}/3rdparty/*)

add_subdirectory(${Stasm_DIR} ${Stasm_DIR}/build)

set(SRC ${SOURCE};${SRC})

include_directories(${Stasm_DIR}/stasm)
link_directories(${Stasm_DIR}/build)

set(Stasm4_LIBS stasm)
