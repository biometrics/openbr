if(UNIX)
  find_package(Threads REQUIRED)
endif()

add_executable(br br.cpp ${BR_RESOURCES})
target_link_libraries(br openbr ${CMAKE_THREAD_LIBS_INIT})
foreach(QT_DEPENDENCY IN LISTS QT_DEPENDENCIES)
    target_link_libraries(br "Qt6::${QT_DEPENDENCY}")
endforeach()

install(TARGETS br RUNTIME DESTINATION bin)
add_test(NAME br_initialize WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} COMMAND br)
