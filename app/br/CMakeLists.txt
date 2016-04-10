if(UNIX)
  find_package(Threads REQUIRED)
endif()

add_executable(br br.cpp ${BR_RESOURCES})
target_link_libraries(br openbr ${CMAKE_THREAD_LIBS_INIT})
qt5_use_modules(br ${QT_DEPENDENCIES})

install(TARGETS br RUNTIME DESTINATION bin)
add_test(NAME br_initialize WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} COMMAND br)

