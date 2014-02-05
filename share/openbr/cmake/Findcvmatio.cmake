set(CVMATIO_DIR "${BR_THIRDPARTY_DIR}/cvmatio")
if(NOT EXISTS ${CVMATIO_DIR})
  # download source from github
  execute_process(COMMAND "git" "clone" "https://github.com/hbristow/cvmatio.git" WORKING_DIRECTORY ${BR_THIRDPARTY_DIR})
else()
  # update the source
  execute_process(COMMAND "git" "pull" WORKING_DIRECTORY ${CVMATIO_DIR})
endif()
add_subdirectory(${CVMATIO_DIR})
include_directories(${CVMATIO_DIR}/include)
set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${CVMATIO_DIR}/src/MatlabIO.cpp)
link_directories(${CMAKE_BINARY_DIR}/3rdparty/cvmatio/src)
set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} cvmatio)
