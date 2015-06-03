set(BR_WITH_EXAMPLE OFF CACHE BOOL "Build Example application.")

if(${BR_WITH_EXAMPLE})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

aux_source_directory(. SRC)

# Build executable
if(APPLE)
  set_source_files_properties(${NATIVE_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
  add_executable(Example MACOSX_BUNDLE ${NATIVE_ICON} ${SRC})
  set_target_properties(Example PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_INSTALL_PREFIX}/Info.plist")
else()
  add_executable(Example WIN32 ${SRC})
endif()

qt5_use_modules(Example ${QT_DEPENDENCIES})

target_link_libraries(Example openbr ${Qt5Core_QTMAIN_LIBRARIES})

install(TARGETS Example RUNTIME DESTINATION bin BUNDLE DESTINATION .)

file(GLOB HEADERS *.h)
install(FILES ${HEADERS} DESTINATION include/example)

# Install openbr.bib so OpenBR can locate it's SDK (or so it thinks)
install(FILES ${CMAKE_SOURCE_DIR}/share/openbr/openbr.bib DESTINATION share/openbr/)

# Make OSX bundle
if(APPLE)
  set(BUNDLE_DIR "${CMAKE_INSTALL_PREFIX}/Example.app")

  install(FILES ${_qt5Core_install_prefix}/plugins/platforms/libqcocoa.dylib
        DESTINATION ${BUNDLE_DIR}/Contents/MacOS/platforms)

  install(FILES ${CMAKE_INSTALL_PREFIX}/bin/br
          DESTINATION ${BUNDLE_DIR}/Contents/MacOS
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE GROUP_WRITE WORLD_WRITE OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)

  set(APPS "${BUNDLE_DIR}/Contents/MacOS/br")
  set(LIBS "platforms/libqcocoa.dylib")

  # To simplify things, I'm assuming that OpenCV is installed in /usr/local/
  set(DIRS "/usr/local/" "${CMAKE_INSTALL_PREFIX}/lib" ${_qt5Core_install_prefix})   # directories to search for prerequisites

  install(CODE "
   include(BundleUtilities)
   fixup_bundle(\"${APPS}\"   \"${LIBS}\"   \"${DIRS}\")
   " DESTINATION bin COMPONENT Runtime)

  # Just put the installed files into the bundle (there's a less clunky way to do this)
  install(DIRECTORY ${CMAKE_INSTALL_PREFIX}/share
          DESTINATION ${BUNDLE_DIR}/Contents/MacOS)
endif()

if (BR_PACKAGE_THIRDPARTY)

set(CPACK_PACKAGE_NAME "Example")
set(CPACK_PACKAGE_VENDOR “Company”)
set(CPACK_PACKAGE_DESCRIPTION "Example")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION_MAJOR 1)
set(CPACK_PACKAGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0)
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(SHARE_DIR “${OPENBR_DIR}/share/example”)

# Define resources
if(WIN32)
  set(NATIVE_ICON "${OPENBR_DIR}/icons/example.ico")
elseif(APPLE)
  set(CPACK_PACKAGE_ICON "${OPENBR_DIR}/icons/example.png")
  set(NATIVE_ICON "${OPENBR_DIR}/icons/example.icns")
else()
  set(CPACK_PACKAGE_ICON "${OPENBR_DIR}/icons/example.png")
  set(NATIVE_ICON "${OPENBR_DIR}/icons/example.png")
endif()

set(CPACK_PACKAGE_EXECUTABLES “Example” "Example")
set(CPACK_CREATE_DESKTOP_LINKS "Example")
if(CMAKE_HOST_WIN32)
  set(CPACK_NSIS_MUI_ICON ${NATIVE_ICON})
  set(CPACK_NSIS_MUI_UNIICON ${NATIVE_ICON})
  if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
  endif()
  set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\ Example.exe")
  set(CPACK_NSIS_EXECUTABLES_DIRECTORY "bin\\\\")
  set(CPACK_CREATE_DESKTOP_LINKS "Example")
  set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "SetOutPath \\\"$INSTDIR\\\\bin\\\"\\nCreateShortCut \\\"$DESKTOP\\\\Example.lnk\\\" \\\"$INSTDIR\\\\bin\\\\ Example.exe\\\"\\nCreateShortCut \\\"$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\Example.lnk\\\" \\\"$INSTDIR\\\\bin\\\\Example.exe\\\"")
  set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "Delete \\\"$DESKTOP\\\\Example.lnk\\\"")
elseif(CMAKE_HOST_APPLE)
  configure_file("${CMAKE_SOURCE_DIR}/README.md" "README.txt" COPYONLY)
  set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_BINARY_DIR}/README.txt")
  set(CPACK_BUNDLE_NAME ${CPACK_PACKAGE_NAME})
  set(CPACK_BUNDLE_ICON example)
  set(MACOSX_BUNDLE_ICON_FILE example)
  configure_file("${SHARE_DIR}/Info.plist.in" "${CMAKE_INSTALL_PREFIX}/Info.plist")
else()
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER “your.email@gmail.com")
endif()
endif()
endif()

include(CPack)
