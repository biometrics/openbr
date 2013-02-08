if(UNIX)
  find_package(Threads REQUIRED)
endif()

add_executable(br br.cpp ${BR_RESOURCES})
target_link_libraries(br openbr ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS br RUNTIME DESTINATION bin)
