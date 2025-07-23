include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(br-gui WIN32 br-gui.cpp ${BR_RESOURCES})

foreach(QT_DEPENDENCY IN LISTS QT_DEPENDENCIES)
  target_link_libraries(br-gui "Qt6::${QT_DEPENDENCY}")
endforeach()

target_link_libraries(br-gui openbr ${BR_THIRDPARTY_LIBS} opencv_core)
install(TARGETS br-gui RUNTIME DESTINATION bin)
