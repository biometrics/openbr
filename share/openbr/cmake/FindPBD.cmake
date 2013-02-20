# ================================================================
# The PartsBasedDetector CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(PBD REQUIRED)
#   target_link_libraries(MY_TARGET ${PBD_LIBS})
# ================================================================

find_path(PBD_DIR include/Candidate.hpp ${CMAKE_SOURCE_DIR}/3rdparty/*)
include_directories(${PBD_DIR}/include)
link_directories(${PBD_DIR}/lib)

find_package(Boost COMPONENTS system serialization REQUIRED)
message(${Boost_INCLUDE_DIR})
message(${Boost_LIBRARIES})
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})
set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${Boost_LIBRARIES})

set(PBD_LIBS PartsBasedDetector)
