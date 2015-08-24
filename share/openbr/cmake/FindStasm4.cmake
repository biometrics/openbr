# ================================================================
# The Stasm CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(Stasm4 REQUIRED)
#   target_link_libraries(MY_TARGET ${Stasm4_LIBS})
# ================================================================

find_path(Stasm_DIR stasm/stasm_lib.h ${PROJECT_SOURCE_DIR}/3rdparty/* NO_DEFAULT_PATH)
mark_as_advanced(Stasm_DIR)
include_directories(${Stasm_DIR}/stasm)
include_directories(${Stasm_DIR}/stasm/MOD_1)
file(GLOB Stasm_SRC "${Stasm_DIR}/stasm/*.cpp")
file(GLOB Stasm_SRC ${Stasm_SRC} "${Stasm_DIR}/stasm/MOD_1/*.cpp")
