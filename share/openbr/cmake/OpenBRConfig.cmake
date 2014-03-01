# ================================================================
#  The OpenBR CMake configuration file
#
#  Usage from an external project:
#    In your CMakeLists.txt, add these lines:
#
#    find_package(OpenBR REQUIRED)
#    target_link_libraries(MY_TARGET_NAME ${OPENBR_LIBS})
# ================================================================

find_path(OPENBR_DIR include/openbr/openbr.h)
include_directories(${OPENBR_DIR}/include)
link_directories(${OPENBR_DIR}/lib)
set(OPENBR_LIBS "openbr")
