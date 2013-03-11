if(UNIX)
  find_package(Threads REQUIRED)
endif()

add_executable(br br.cpp ${BR_RESOURCES})
target_link_libraries(br openbr ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS br RUNTIME DESTINATION bin)

add_test(NAME br_initialize WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND br)
add_test(NAME br_objects WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND br -objects)
add_test(NAME br_draw_face_detection WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND br -algorithm DrawFaceDetection -enroll ../data/family.jpg)
