include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(br-docs br-docs.cpp ${BR_RESOURCES})

foreach(QT_DEPENDENCY IN LISTS QT_DEPENDENCIES)
  target_link_libraries(br-docs "Qt6::${QT_DEPENDENCY}")
endforeach()

target_link_libraries(br-docs openbr ${BR_THIRDPARTY_LIBS} opencv_core)
install(TARGETS br-docs RUNTIME DESTINATION bin)
