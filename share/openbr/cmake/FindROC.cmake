# ================================================================
# The ROC CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(ROC REQUIRED)
#   target_link_libraries(MY_TARGET ${ROC_LIBS})
# ================================================================

find_path(ROC_DIR include/roc.h ${CMAKE_SOURCE_DIR}/3rdparty/*)
include_directories(${ROC_DIR}/include)
link_directories(${ROC_DIR}/lib)
set(ROC_LIBS roc)
