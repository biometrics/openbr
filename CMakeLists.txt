project(openbr)
cmake_minimum_required(VERSION 2.8.9)

# Global settings
set(BR_SHARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/share/openbr")
set(BR_SCRIPTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
set(CMAKE_AUTOMOC ON)
set(CPACK_PACKAGE_NAME "OpenBR")
set(CPACK_PACKAGE_VENDOR "OpenBiometrics")
set(CPACK_PACKAGE_DESCRIPTION "Open Source Biometric Recognition")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
if(NOT DEFINED CPACK_PACKAGE_VERSION_MAJOR)
  set(CPACK_PACKAGE_VERSION_MAJOR 1)
endif()
if(NOT DEFINED CPACK_PACKAGE_VERSION_MINOR)
  set(CPACK_PACKAGE_VERSION_MINOR 1)
endif()
if(NOT DEFINED CPACK_PACKAGE_VERSION_PATCH)
  set(CPACK_PACKAGE_VERSION_PATCH 0)
endif()
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CMAKE_MODULE_PATH "${BR_SHARE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(PACKAGE_YEAR 2013)

if(${CMAKE_VERSION} VERSION_EQUAL 2.8.11)
  cmake_policy(SET CMP0020 OLD)
endif()
if(${CMAKE_VERSION} VERSION_GREATER 2.8.11)
  cmake_policy(SET CMP0020 OLD)
endif()

if(${CMAKE_VERSION} VERSION_EQUAL 2.8.12)
  cmake_policy(SET CMP0022 OLD)
endif()
if(${CMAKE_VERSION} VERSION_GREATER 2.8.12)
  cmake_policy(SET CMP0022 OLD)
endif()

if(${CMAKE_VERSION} VERSION_GREATER 3)
  cmake_policy(SET CMP0043 NEW)
endif()

if(${CMAKE_SIZEOF_VOID_P} MATCHES 8)
  set(BITNESS 64)
else()
  set(BITNESS 32)
endif()

include(ExternalProject)

# Define resources
if(WIN32)
  set(CPACK_PACKAGE_ICON "${BR_SHARE_DIR}\\\\openbr_small.png")
  set(NATIVE_ICON "${BR_SHARE_DIR}/openbr.ico")
elseif(APPLE)
  set(CPACK_PACKAGE_ICON "${BR_SHARE_DIR}/openbr_small.png")
  set(NATIVE_ICON "${BR_SHARE_DIR}/openbr.icns")
else()
  set(CPACK_PACKAGE_ICON "${BR_SHARE_DIR}/openbr_small.png")
  set(NATIVE_ICON "${BR_SHARE_DIR}/openbr.png")
endif()

if(WIN32)
  configure_file(${BR_SHARE_DIR}/resources.rc.in resources.rc)
  set(BR_RESOURCES ${CMAKE_CURRENT_BINARY_DIR}/resources.rc)
endif()

# Build options
option(BR_EMBEDDED "Limit software dependencies")
if(BR_EMBEDDED)
  add_definitions(-DBR_EMBEDDED)
endif()

# Find Qt
set(QT_DEPENDENCIES ${QT_DEPENDENCIES} Concurrent Core)
if(NOT BR_EMBEDDED)
  set(QT_DEPENDENCIES ${QT_DEPENDENCIES} Gui Network Sql Widgets Xml)
endif()

find_package(Qt5 COMPONENTS ${QT_DEPENDENCIES} REQUIRED)

set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${Qt5Core_QTMAIN_LIBRARIES})
if(IOS)
  list(APPEND BR_THIRDPARTY_LIBS "${_qt5Core_install_prefix}/lib/libqtpcre2.a")
endif()

# Find OpenCV
find_package(OpenCV 2.4.5 REQUIRED)
if(IOS)
  file(GLOB OPENCV_IOS_LIBS "${OpenCV_DIR}/../../lib/*.a")
  list(APPEND BR_THIRDPARTY_LIBS ${OPENCV_IOS_LIBS})
  list(APPEND BR_THIRDPARTY_LIBS "-framework Foundation -framework UIKit -framework Security -framework MobileCoreServices -lc++")
else()
  set(OPENCV_DEPENDENCIES opencv_core opencv_highgui opencv_imgproc opencv_ml)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${OPENCV_DEPENDENCIES})
  if(APPLE)
    list(APPEND BR_THIRDPARTY_LIBS "-framework Accelerate")
  endif()
endif()

# Find Alphanum
find_package(Alphanum REQUIRED)

# Find cvmatio (if using it)
set(BR_WITH_CVMATIO OFF CACHE BOOL "Build with cvmatio library to read Matlab data files (required to use Caltech Pedestrians dataset)")
if(${BR_WITH_CVMATIO})
  find_package(cvmatio REQUIRED)
  add_definitions(-DCVMATIO)
  add_subdirectory(${CVMATIO_DIR})
  include_directories(${CVMATIO_DIR}/include)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${CVMATIO_DIR}/src/MatlabIO.cpp)
  link_directories(${CVMATIO_LIB_DIR})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} cvmatio)
endif()

# Compiler flags
if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-overflow -Wno-terminate -Wno-sign-compare -Wno-unused-local-typedefs -Wno-int-in-bool-context -Wno-comment -Wno-unknown-pragmas -fvisibility=hidden -fno-omit-frame-pointer")
  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib ${_qt5Core_install_prefix}/lib)
  if(NOT APPLE)
    if(${CMAKE_CXX_COMPILER} STREQUAL "/opt/intel/bin/icpc")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-intel -wd2196")
    endif()
  endif()
else()
  if(MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,--enable-auto-import") # Fixes a linker warning
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--enable-auto-import")
  elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /wd4018 /wd4244 /wd4267 /wd4305 /wd4308 /wd4307 /wd4554 /wd4996 /w34100 /nologo /MP")
  endif()
endif()

# Enable Testing
include(CppcheckTargets)
include(CTest)
enable_testing()
if(BUILD_TESTING)
  set(BUILDNAME "${BUILDNAME}" CACHE STRING "Name of build on the dashboard")
  mark_as_advanced(BUILDNAME)
endif()

# Look for extensions to OpenBR
set(BR_THIRDPARTY_PLUGINS_DIR CACHE PATH "")
set(BR_THIRDPARTY_APPS_DIR CACHE PATH "")
mark_as_advanced(BR_THIRDPARTY_PLUGINS_DIR)
mark_as_advanced(BR_THIRDPARTY_APPS_DIR)

if(BR_THIRDPARTY_DIR)
  if(EXISTS "${BR_THIRDPARTY_DIR}/src/src.cmake")
    include("${BR_THIRDPARTY_DIR}/src/src.cmake")
  endif()
  if(EXISTS "${BR_THIRDPARTY_DIR}/plugins")
    set(BR_THIRDPARTY_PLUGINS_DIR ${BR_THIRDPARTY_PLUGINS_DIR} "${BR_THIRDPARTY_DIR}/plugins")
  endif()
  if(EXISTS "${BR_THIRDPARTY_DIR}/app")
    set(BR_THIRDPARTY_APPS_DIR    ${BR_THIRDPARTY_APPS_DIR}    "${BR_THIRDPARTY_DIR}/app")
  endif()
else()
  set(BR_THIRDPARTY_DIR "" CACHE PATH "Path to proprietary software")
  mark_as_advanced(BR_THIRDPARTY_DIR)
endif()

# Download the models
ExternalProject_Add(models
                    URL http://github.com/biometrics/openbr/releases/download/v1.1.0/models.tar.gz
                    URL_MD5 26cf71abd48cec8d7b2abf3e8f4ebfc4
                    SOURCE_DIR "${PROJECT_SOURCE_DIR}/share/openbr/models"
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND "")

# Build the SDK
include_directories(BEFORE .) # Find the local headers first
add_subdirectory(openbr)

# Build applications
add_subdirectory(app)
foreach(DIR ${BR_THIRDPARTY_APPS_DIR})
  get_filename_component(FNAME ${DIR} NAME)
  # DIR could be out-of-tree so we must specify the build directory
  add_subdirectory(${DIR} 3rdparty_apps/${FNAME})
endforeach()

# Install
option(BR_INSTALL_SHARE "Install share/openbr files." ON)
if(${BR_INSTALL_SHARE})
  install(FILES LICENSE.txt README.md DESTINATION share/openbr)
  install(DIRECTORY share DESTINATION .)
  install(DIRECTORY ${BR_THIRDPARTY_SHARE} DESTINATION share)
endif()

# install brpy
option(BR_INSTALL_BRPY "Install brpy, the Python wrapper to the C API (requires Python)")
if(${BR_INSTALL_BRPY})
  find_package(PythonInterp REQUIRED)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import site, sys; sys.stdout.write(site.getsitepackages()[-1])" OUTPUT_VARIABLE PYTHON_SITE_DIR)
  install(DIRECTORY ${BR_SCRIPTS_DIR}/brpy DESTINATION ${PYTHON_SITE_DIR})
endif()

# Package
if(NOT BR_PACKAGE_THIRDPARTY)
  set(CPACK_PACKAGE_EXECUTABLES "OpenBR" "OpenBR")
  set(CPACK_CREATE_DESKTOP_LINKS "OpenBR")
  if(CMAKE_HOST_WIN32)
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_MUI_ICON ${NATIVE_ICON})
    set(CPACK_NSIS_MUI_UNIICON ${NATIVE_ICON})
    set(CPACK_NSIS_MENU_LINKS "doc/html/index.html" "Documentation")
    if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
      set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
    endif()
    set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\br-gui.exe")
  elseif(CMAKE_HOST_APPLE)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/README.md" "README.txt" COPYONLY)
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_BINARY_DIR}/README.txt")
    set(CPACK_BUNDLE_NAME ${CPACK_PACKAGE_NAME})
    set(CPACK_BUNDLE_ICON ${NATIVE_ICON})
    set(CPACK_BUNDLE_PLIST ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)
    set(CPACK_BUNDLE_STARTUP_COMMAND ${BR_SHARE_DIR}/bundle.sh)
    configure_file(${BR_SHARE_DIR}/Info.plist.in Info.plist)
  else()
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "josh.klontz@gmail.com")
  endif()

  include(CPack)
endif()
