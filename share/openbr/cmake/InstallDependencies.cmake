set(BR_INSTALL_DEPENDENCIES OFF CACHE BOOL "Install runtime dependencies.")

# OpenCV Libs
function(install_opencv_library lib)
  if(${BR_INSTALL_DEPENDENCIES})
    if(CMAKE_HOST_WIN32)
      if(${CMAKE_BUILD_TYPE} MATCHES Debug)
        set(BR_INSTALL_DEPENDENCIES_SUFFIX "d")
      endif()
      if(NOT MSVC)
        set(BR_INSTALL_DEPENDENCIES_PREFIX "lib")
      endif()
      install(FILES ${OpenCV_DIR}/bin/${BR_INSTALL_DEPENDENCIES_PREFIX}${lib}${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}${BR_INSTALL_DEPENDENCIES_SUFFIX}.dll DESTINATION bin)
    else()
      set(OpenCV_LIB_DIR "/usr/local/lib")
      install(FILES ${OpenCV_LIB_DIR}/lib${lib}.${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.${OpenCV_VERSION_PATCH}${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION lib)
      install(FILES ${OpenCV_LIB_DIR}/lib${lib}.${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION lib)
      install(FILES ${OpenCV_LIB_DIR}/lib${lib}${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION lib)
    endif()
  endif()
endfunction()

function(install_opencv_libraries libs)
  foreach(lib ${${libs}})
    install_opencv_library(${lib})
  endforeach()
endfunction()

# Qt Libs
function(install_qt_library lib)
  if(${BR_INSTALL_DEPENDENCIES})
    if(CMAKE_HOST_WIN32)
      if(${CMAKE_BUILD_TYPE} MATCHES Debug)
        set(BR_INSTALL_DEPENDENCIES_SUFFIX "d")
      endif()
      install(FILES ${_qt5Core_install_prefix}/lib/Qt5${lib}${BR_INSTALL_DEPENDENCIES_SUFFIX}.dll DESTINATION bin)
    elseif(CMAKE_HOST_APPLE)
      install(DIRECTORY ${_qt5Core_install_prefix}/lib/Qt${lib}.framework DESTINATION lib)
    else()
      install(FILES ${_qt5Core_install_prefix}/lib/libQt5${lib}.so.5.${Qt5Core_VERSION_MINOR}.${Qt5Core_VERSION_PATCH} DESTINATION lib)
      install(FILES ${_qt5Core_install_prefix}/lib/libQt5${lib}.so.5.${Qt5Core_VERSION_MINOR} DESTINATION lib)
      install(FILES ${_qt5Core_install_prefix}/lib/libQt5${lib}.so.5 DESTINATION lib)
      install(FILES ${_qt5Core_install_prefix}/lib/libQt5${lib}.so DESTINATION lib)
    endif()
  endif()
endfunction()

function(install_qt_libraries libs)
  foreach(lib ${${libs}})
    install_qt_library(${lib})
  endforeach()
endfunction()

# Qt Plugins
function(install_qt_imageformats)
  if(${BR_INSTALL_DEPENDENCIES})
    install(FILES ${QT_QGIF_PLUGIN_RELEASE}
                  ${QT_QICO_PLUGIN_RELEASE}
                  ${QT_QJPEG_PLUGIN_RELEASE}
                  ${QT_QMNG_PLUGIN_RELEASE}
                  ${QT_QSVG_PLUGIN_RELEASE}
                  ${QT_QTIFF_PLUGIN_RELEASE}
            DESTINATION bin/imageformats)
  endif()
endfunction()

# Qt Other
function(install_qt_misc)
  if(MSVC)
    if(${CMAKE_BUILD_TYPE} MATCHES Debug)
      set(BR_INSTALL_DEPENDENCIES_SUFFIX "d")
    endif()
    install(FILES ${_qt5Core_install_prefix}/bin/libGLESv2${BR_INSTALL_DEPENDENCIES_SUFFIX}.dll DESTINATION bin)
    # install(FILES ${_qt5Core_install_prefix}/bin/icuin49${BR_INSTALL_DEPENDENCIES_SUFFIX}.dll DESTINATION bin)
  endif()
endfunction()

# Compiler libraries
function(install_compiler_libraries)
  include(InstallRequiredSystemLibraries)
  if(${BR_INSTALL_DEPENDENCIES} AND MINGW)
    set(MINGW_DIR "MINGW_DIR-NOTFOUND" CACHE PATH "MinGW Path")
    get_filename_component(MINGW_DIR ${CMAKE_CXX_COMPILER} PATH)
    install(FILES ${MINGW_DIR}/libgcc_s_sjlj-1.dll ${MINGW_DIR}/libstdc++-6.dll DESTINATION bin)
  endif()
endfunction()

# R runtime
function(install_r_runtime)
  if(WIN32)
    find_path(R_DIR bin/Rscript.exe "C:/Program Files/R/*")
    install(DIRECTORY ${R_DIR}/ DESTINATION R)
  endif()
endfunction()
