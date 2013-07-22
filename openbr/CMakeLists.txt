# Create version.h
configure_file(${BR_SHARE_DIR}/version.h.in version.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Collect source files
aux_source_directory(. SRC)
aux_source_directory(core BR_CORE)
include(plugins/plugins.cmake)

# Optional GUI module
if(NOT ${BR_EMBEDDED})
  aux_source_directory(gui BR_GUI)
  qt5_add_resources(BR_ICONS icons/icons.qrc)
  file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/gui/*.h)
  install(FILES ${HEADERS} DESTINATION include/openbr/gui)
endif()

add_library(openbr SHARED ${SRC} ${BR_CORE} ${BR_GUI} ${BR_ICONS} ${BR_THIRDPARTY_SRC} ${BR_RESOURCES} ${NATURALSTRINGCOMPARE_SRC})
qt5_use_modules(openbr ${QT_DEPENDENCIES})
set_target_properties(openbr PROPERTIES
                             DEFINE_SYMBOL BR_LIBRARY
                             VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}
                             SOVERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}
                             LINK_INTERFACE_LIBRARIES "")
target_link_libraries(openbr ${BR_THIRDPARTY_LIBS})
add_cppcheck(openbr)

# Install
install(TARGETS openbr
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
file(GLOB HEADERS *.h)
install(FILES ${HEADERS} DESTINATION include/openbr)
include(InstallDependencies)
install_compiler_libraries()
install_qt_libraries(QT_DEPENDENCIES)
install_qt_imageformats()
install_qt_misc()
install_opencv_libraries(OPENCV_DEPENDENCIES)
