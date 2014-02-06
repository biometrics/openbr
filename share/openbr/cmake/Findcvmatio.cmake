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
set(CVMATIO_LIB_DIR ${CMAKE_BINARY_DIR}/3rdparty/cvmatio/src)
link_directories(${CVMATIO_LIB_DIR})
set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} cvmatio)
# because the project's cmake installs to the project folder,
# install(TARGETS cvmatio DESTINATION lib) doesn't work
# (i get error "target doesn't exist in this directory")
# so yeah, this won't work in windows
install(FILES ${CVMATIO_LIB_DIR}/libcvmatio.dylib DESTINATION lib)
