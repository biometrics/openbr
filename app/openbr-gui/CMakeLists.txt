if(NOT ${BR_EMBEDDED})
  include_directories(.)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})

  file(GLOB SRC *.cpp)
  file(GLOB HEADERS *.h)
  qt4_add_resources(ICONS icons/icons.qrc) # Don't forget to modify initialize.cpp

  add_library(openbr-gui SHARED ${SRC} ${ICONS} ${BR_WINDOWS_RESOURCES})
  set_target_properties(openbr-gui PROPERTIES
                                   DEFINE_SYMBOL BR_LIBRARY_GUI
                                   VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}
                                   SOVERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}
                                   LINK_INTERFACE_LIBRARIES ""
                                   AUTOMOC TRUE)
  target_link_libraries(openbr-gui openbr ${QT_LIBRARIES} ${OpenCV_LIBS})
  add_cppcheck(openbr-gui)

  install(FILES ${HEADERS} DESTINATION include/openbr-gui)
  install(TARGETS openbr-gui RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
endif()
