set(CVMATIO_DIR "${BR_THIRDPARTY_DIR}/cvmatio")
if(NOT EXISTS ${CVMATIO_DIR})
  # download source from github
  execute_process(COMMAND "git" "clone" "https://github.com/hbristow/cvmatio.git" WORKING_DIRECTORY ${BR_THIRDPARTY_DIR})
else()
  # update the source
  execute_process(COMMAND "git" "pull" WORKING_DIRECTORY ${CVMATIO_DIR})
endif()
set(CVMATIO_LIB_DIR ${CMAKE_BINARY_DIR}/3rdparty/cvmatio/src)
