set(BR_INSTALL_DEPENDENCIES OFF CACHE BOOL "Install runtime dependencies.")

# OpenCV Libs
function(install_opencv_library lib)
  if(${BR_INSTALL_DEPENDENCIES} AND ${OpenCV_SHARED})
    if(CMAKE_HOST_WIN32)
      if(${CMAKE_BUILD_TYPE} MATCHES Debug)
        set(BR_INSTALL_DEPENDENCIES_SUFFIX "d")
      endif()
      if(NOT MSVC)
        set(BR_INSTALL_DEPENDENCIES_PREFIX "lib")
      endif()
      list(GET OpenCV_LIB_DIR 0 cv_lib_stripped)
      install(FILES ${cv_lib_stripped}/../bin/${BR_INSTALL_DEPENDENCIES_PREFIX}${lib}${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}${BR_INSTALL_DEPENDENCIES_SUFFIX}.dll DESTINATION bin)
    elseif(CMAKE_HOST_APPLE)
      set(OpenCV_LIB_DIR "/usr/local/lib")
      install(FILES ${OpenCV_LIB_DIR}/lib${lib}.${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.${OpenCV_VERSION_PATCH}${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION lib)
      install(FILES ${OpenCV_LIB_DIR}/lib${lib}.${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION lib)
      install(FILES ${OpenCV_LIB_DIR}/lib${lib}${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION lib)
    else()
      set(OpenCV_LIB_DIR "/usr/local/lib")
      install(FILES ${OpenCV_LIB_DIR}/lib${lib}${CMAKE_SHARED_LIBRARY_SUFFIX}.${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.${OpenCV_VERSION_PATCH} DESTINATION lib)
      install(FILES ${OpenCV_LIB_DIR}/lib${lib}${CMAKE_SHARED_LIBRARY_SUFFIX}.${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR} DESTINATION lib)
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
    if(ANDROID)
      set(FILE ${_qt5Core_install_prefix}/lib/libQt5${lib}.so)
      if(NOT EXISTS ${FILE})
        if(ANDROID_ABI STREQUAL "arm64-v8a")
          set(ABI "arm64-v8a")
        else()
          set(ABI "armeabi-v7a")
        endif()
        set(FILE ${_qt5Core_install_prefix}/lib/libQt5${lib}_${ABI}.so)
      endif()
      install(FILES ${FILE} DESTINATION lib)
    elseif(CMAKE_HOST_WIN32)
      if(${CMAKE_BUILD_TYPE} MATCHES Debug)
        set(BR_INSTALL_DEPENDENCIES_SUFFIX "d")
      endif()
      install(FILES ${_qt5Core_install_prefix}/bin/Qt5${lib}${BR_INSTALL_DEPENDENCIES_SUFFIX}.dll DESTINATION bin)
    elseif(CMAKE_HOST_APPLE)
      if (EXISTS "${_qt5Core_install_prefix}/lib/Qt${lib}.framework")
        install(DIRECTORY ${_qt5Core_install_prefix}/lib/Qt${lib}.framework DESTINATION lib)
      else()
        install(FILES ${_qt5Core_install_prefix}/lib/libQt5${lib}.5.${Qt5_VERSION_MINOR}.${Qt5_VERSION_PATCH}.dylib DESTINATION lib)
        install(FILES ${_qt5Core_install_prefix}/lib/libQt5${lib}.5.${Qt5_VERSION_MINOR}.dylib DESTINATION lib)
        install(FILES ${_qt5Core_install_prefix}/lib/libQt5${lib}.5.dylib DESTINATION lib)
        install(FILES ${_qt5Core_install_prefix}/lib/libQt5${lib}.dylib DESTINATION lib)
      endif()
    else()
      set(Qt5_LIB_DIR "${Qt5_DIR}/../..")
      if (NOT EXISTS "${Qt5_LIB_DIR}/libQt5${lib}.so")
        set(Qt5_LIB_DIR "${_qt5Core_install_prefix}/lib")
      endif()
      install(FILES ${Qt5_LIB_DIR}/libQt5${lib}.so.5.${Qt5_VERSION_MINOR}.${Qt5_VERSION_PATCH} DESTINATION lib)
      install(FILES ${Qt5_LIB_DIR}/libQt5${lib}.so.5.${Qt5_VERSION_MINOR} DESTINATION lib)
      install(FILES ${Qt5_LIB_DIR}/libQt5${lib}.so.5 DESTINATION lib)
      install(FILES ${Qt5_LIB_DIR}/libQt5${lib}.so DESTINATION lib)
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
    set(Qt5_PLUGIN_DIR "${Qt5_DIR}/../../qt5/plugins")
    if (NOT EXISTS ${Qt5_PLUGIN_DIR})
      set(Qt5_PLUGIN_DIR "${Qt5_DIR}/../../../plugins")
    endif()
    set(IMAGE_FORMATS_DIR "${Qt5_PLUGIN_DIR}/imageformats")
    if(ANDROID)
      set(INSTALL_DEPENDENCIES_PREFIX "lib")
      set(INSTALL_DEPENDENCIES_EXTENSION ".so")
    elseif(CMAKE_HOST_WIN32)
      set(INSTALL_DEPENDENCIES_PREFIX "")
      set(INSTALL_DEPENDENCIES_EXTENSION ".dll")
    elseif(CMAKE_HOST_APPLE)
      set(INSTALL_DEPENDENCIES_PREFIX "lib")
      set(INSTALL_DEPENDENCIES_EXTENSION ".dylib")
    else()
      set(INSTALL_DEPENDENCIES_PREFIX "lib")
      set(INSTALL_DEPENDENCIES_EXTENSION ".so")
    endif()

    foreach (IMGPLUGIN qgif qico qjpeg qmng qsvg qtga qtiff qwbmp)
      set(IMGFILE "${IMAGE_FORMATS_DIR}/${INSTALL_DEPENDENCIES_PREFIX}${IMGPLUGIN}${INSTALL_DEPENDENCIES_EXTENSION}")
      if(EXISTS ${IMGFILE})
        install(FILES ${IMGFILE}
                DESTINATION bin/imageformats)
      endif()
    endforeach()
  endif()
endfunction()

function(install_qt_audio)
  if(${BR_INSTALL_DEPENDENCIES})
    set(Qt5_PLUGIN_DIR "${Qt5_DIR}/../../qt5/plugins")
    if (NOT EXISTS ${Qt5_PLUGIN_DIR})
      set(Qt5_PLUGIN_DIR "${Qt5_DIR}/../../../plugins")
    endif()
    if(CMAKE_HOST_WIN32)
      install(FILES ${AUDIO_DIR}/qtaudio_windows.dll DESTINATION bin/audio)
    elseif(CMAKE_HOST_APPLE)
      install(FILES ${AUDIO_DIR}/libqtaudio_coreaudio.dylib DESTINATION bin/audio)
    else()
      install(FILES ${AUDIO_DIR}/libqtaudio_alsa.so DESTINATION bin/audio)
      install(FILES ${AUDIO_DIR}/libqtmedia_pulse.so DESTINATION bin/audio)
    endif()
  endif()
endfunction()

function(install_qt_sql)
  if(${BR_INSTALL_DEPENDENCIES})
    set(Qt5_PLUGIN_DIR "${Qt5_DIR}/../../qt5/plugins")
    if (NOT EXISTS ${Qt5_PLUGIN_DIR})
      set(Qt5_PLUGIN_DIR "${Qt5_DIR}/../../../plugins")
    endif()
    set(SQL_DIR "${Qt5_PLUGIN_DIR}/sqldrivers")
    if(CMAKE_HOST_WIN32)
      install(FILES ${SQL_DIR}/qsqlite.dll DESTINATION bin/sqldrivers)
    elseif(CMAKE_HOST_APPLE)
      install(FILES ${SQL_DIR}/libqsqlite.dylib DESTINATION bin/sqldrivers)
    else()
      install(FILES ${SQL_DIR}/libqsqlite.so DESTINATION bin/sqldrivers)
    endif()
  endif()
endfunction()

function(install_qt_styles)
  if(${BR_INSTALL_DEPENDENCIES})
    set(Qt5_PLUGIN_DIR "${Qt5_DIR}/../../qt5/plugins")
    if (NOT EXISTS ${Qt5_PLUGIN_DIR})
      set(Qt5_PLUGIN_DIR "${Qt5_DIR}/../../../plugins")
    endif()
    set(STYLES_DIR "${Qt5_PLUGIN_DIR}/styles")
    if(CMAKE_HOST_WIN32)
      # Nothing to do
    elseif(CMAKE_HOST_APPLE)
      install(FILES ${STYLES_DIR}/libqmacstyle.dylib DESTINATION bin/styles)
    else()
      # Nothing to do
    endif()
  endif()
endfunction()

function(install_qt_platforms)
  if(${BR_INSTALL_DEPENDENCIES})
    if(ANDROID)
      install(FILES ${_qt5Core_install_prefix}/plugins/platforms/android/libqtforandroid.so
              DESTINATION bin/platforms)
    elseif(CMAKE_HOST_WIN32)
      install(FILES ${_qt5Core_install_prefix}/plugins/platforms/qwindows.dll
              DESTINATION bin/platforms)
    elseif(CMAKE_HOST_APPLE)
      install(FILES ${_qt5Core_install_prefix}/plugins/platforms/libqcocoa.dylib
              DESTINATION bin/platforms)
    else()
      set(Qt5_PLUGIN_DIR "${Qt5_DIR}/../../qt5/plugins")
      if (NOT EXISTS ${Qt5_PLUGIN_DIR})
        set(Qt5_PLUGIN_DIR "${Qt5_DIR}/../../../plugins")
      endif()
      install(FILES ${Qt5_PLUGIN_DIR}/platforms/libqlinuxfb.so
              DESTINATION bin/platforms)

      if(EXISTS ${Qt5_PLUGIN_DIR}/platforms/libqxcb.so)
        install_qt_library(XcbQpa)
        install(FILES ${Qt5_PLUGIN_DIR}/platforms/libqxcb.so
                DESTINATION bin/platforms)
      endif()
    endif()
  endif()
endfunction()

# Qt Other
function(install_qt_misc)
  if(MSVC)
    if(${CMAKE_BUILD_TYPE} MATCHES Debug)
      set(BR_INSTALL_DEPENDENCIES_SUFFIX "d")
    endif()
    install(FILES ${_qt5Core_install_prefix}/bin/libGLESv2${BR_INSTALL_DEPENDENCIES_SUFFIX}.dll DESTINATION bin)
    install(FILES ${_qt5Core_install_prefix}/bin/libEGL${BR_INSTALL_DEPENDENCIES_SUFFIX}.dll DESTINATION bin)
    file(GLOB d3dcomp ${_qt5Core_install_prefix}/bin/d3dcompiler_*.dll)
    install(FILES ${d3dcomp} DESTINATION bin)
    install(FILES ${_qt5Core_install_prefix}/plugins/platforms/qwindows${BR_INSTALL_DEPENDENCIES_SUFFIX}.dll DESTINATION bin/platforms)
  elseif(ANDROID)
    install(FILES ${__libstl} DESTINATION lib)
    install(FILES ${_qt5Core_install_prefix}/jar/QtAndroid.jar
            DESTINATION java)
  elseif(UNIX AND NOT APPLE)
    file(GLOB libpng12 "/lib/x86_64-linux-gnu/libpng12.so*")
    install(FILES ${libpng12} DESTINATION lib)
  endif()
endfunction()

# Qt ICU
function(install_qt_icu)
  if(MSVC)
    file(GLOB icudlls ${_qt5Core_install_prefix}/bin/icu*.dll)
    install(FILES ${icudlls} DESTINATION bin)
  elseif(UNIX AND NOT APPLE AND NOT ANDROID)
    set(Qt5_LIB_DIR "${Qt5_DIR}/../..")
    file(GLOB icudlls "${Qt5_LIB_DIR}/libicu*.so*")
    if(NOT icudlls)
      file(GLOB icudlls /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/libicu*.so*)
    endif()
    install(FILES ${icudlls} DESTINATION lib)
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

# FFMPEG
function(install_ffmpeg_help LIB)
  if(LIB)
    string(REGEX REPLACE "\\.[^.]*$" "" STRIPEXT ${LIB})
    file(GLOB LIBS "${STRIPEXT}.*[^a]")
    install(FILES ${LIBS} DESTINATION lib)
  endif()
endfunction()

function(install_ffmpeg)
  if(${BR_INSTALL_DEPENDENCIES})
    if(WIN32)
      list(GET OpenCV_LIB_DIR 0 cv_lib_stripped)
      if(${CMAKE_SIZEOF_VOID_P} MATCHES 8)
        set(FFMPEGSUFFIX "_64")
      else()
        set(FFMPEGSUFFIX "")
      endif()
      set(FFMPEG_LIB "${cv_lib_stripped}/../bin/${BR_INSTALL_DEPENDENCIES_PREFIX}opencv_ffmpeg${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}${FFMPEGSUFFIX}.dll")
      if(EXISTS ${FFMPEG_LIB})
        install(FILES ${FFMPEG_LIB} DESTINATION bin)
      endif()
    else()
      # find_library(AVCODEC avcodec)
      # install_ffmpeg_help(${AVCODEC})

      # find_library(AVUTIL avutil)
      # install_ffmpeg_help(${AVUTIL})

      # find_library(AVFORMAT avformat)
      # install_ffmpeg_help(${AVFORMAT})

      # find_library(SWSCALE swscale)
      # install_ffmpeg_help(${SWSCALE})
    endif()
  endif()
endfunction()
