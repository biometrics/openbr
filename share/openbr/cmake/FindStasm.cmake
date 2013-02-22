# ================================================================
# The Stasm CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(Stasm REQUIRED)
#   target_link_libraries(MY_TARGET ${Stasm_LIBS})
# ================================================================

find_path(Stasm_DIR include/stasm.hpp ${CMAKE_SOURCE_DIR}/3rdparty/*)

include_directories(${Stasm_DIR}/include)
link_directories(${Stasm_DIR}/lib)

set(Stasm_LIBS stasm)
