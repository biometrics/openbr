project (RandomLib)

set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 9)
set (PROJECT_VERSION_PATCH 0)
set (PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
if (PROJECT_VERSION_PATCH GREATER 0)
  set (PROJECT_VERSION "${PROJECT_VERSION}.${PROJECT_VERSION_PATCH}")
endif ()

if (DEFINED CPACK_PACKAGE_VERSION_COUNT)

  # majic (version 0.1.9 and later) invokes cmake defining, e.g.,
  #   -D CPACK_PACKAGE_VERSION=1.9-SNAPSHOT
  #   -D CPACK_PACKAGE_VERSION_COUNT=2
  #   -D CPACK_PACKAGE_VERSION_MAJOR=1
  #   -D CPACK_PACKAGE_VERSION_MINOR=9
  #   -D CPACK_PACKAGE_VERSION_SUFFIX=-SNAPSHOT
  # Check that the first two version numbers are consistent.
  if (CPACK_PACKAGE_VERSION_COUNT EQUAL 2)
    set (CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
  elseif (CPACK_PACKAGE_VERSION_COUNT LESS 2)
    message (FATAL_ERROR "CPACK_PACKAGE_VERSION_COUNT must be 2 or more")
  endif ()
  if (NOT (
        CPACK_PACKAGE_VERSION_MAJOR EQUAL PROJECT_VERSION_MAJOR AND
        CPACK_PACKAGE_VERSION_MINOR EQUAL PROJECT_VERSION_MINOR))
    message (FATAL_ERROR "Inconsistency in CPACK and PROJECT version numbers")
  endif ()
  set (PROJECT_VERSION_PATCH ${CPACK_PACKAGE_VERSION_PATCH})
  set (PROJECT_VERSION ${CPACK_PACKAGE_VERSION})

else ()

  set (CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
  set (CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
  set (CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
  set (CPACK_PACKAGE_VERSION ${PROJECT_VERSION})

endif ()

set (LIBVERSION 1)
set (LIBVERSIONFULL 1.0.9)
string (TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
string (TOUPPER ${PROJECT_NAME} PROJECT_NAME_UPPER)

cmake_minimum_required (VERSION 2.8.4) # This version was released 2011-02-16

# User-settable variables

# (1) COMMON_INSTALL_PATH governs the installation convention.  If it
# is on ON (the Linux default), the installation is to a common
# directory, e.g., /usr/local.  If it is OFF (the Windows default),
# the installation directory contains the package name, e.g.,
# c:/pkg/RandomLib-1.6.  The installation directories for the
# documentation and cmake configuration all depend on the variable
# with deeper paths relative to CMAKE_INSTALL_PREFIX being used when
# it's ON.

if (WIN32)
  option (COMMON_INSTALL_PATH "Use a common installation path for packages" OFF)
else ()
  option (COMMON_INSTALL_PATH "Use a common installation path for packages" ON)
endif ()

# The use of PACKAGE_PATH and INSTALL_PATH is now DEPRECATED.
# (2) PACKAGE_PATH and INSTALL_PATH govern the find_package search
# path and the installation directory.  (find_package is not used by
# RandomLib since it doesn't depend on other packages.  However
# PACKAGE_PATH is used here for uniformity with other packages which
# adopt the same conventions.)
#
# If PACKAGE_PATH is defined, it is prepended to CMAKE_PREFIX_PATH.
#
# If INSTALL_PATH is not specified but PACKAGE_PATH is, then
# INSTALL_PATH is set to
#   ${PACKAGE_PATH}, if COMMON_INSTALL_PATH is ON;
#   ${PACKAGE_PATH}/${PROJECT_NAME}-${PROJECT_VERSION}, otherwise.
#
# If INSTALL_PATH is now defined, then set CMAKE_INSTALL_PREFIX to
# INSTALL_PATH.
#
# Typically, only PACKAGE_PATH needs to be specified, e.g.,
# cmake -D PACKAGE_PATH=/opt .. (on Linux)
#   => CMAKE_PREFIX_PATH=/opt   CMAKE_INSTALL_PREFIX=/opt
# cmake -D PACKAGE_PATH=C:/pkg .. (on Windows)
#   => CMAKE_PREFIX_PATH=C:/pkg CMAKE_INSTALL_PREFIX=C:/pkg/RandomLib-1.6

if (PACKAGE_PATH)
  set (CMAKE_PREFIX_PATH ${PACKAGE_PATH} ${CMAKE_PREFIX_PATH})
  message (STATUS "CMAKE_PREFIX_PATH set to ${CMAKE_PREFIX_PATH}")
endif ()

if (NOT INSTALL_PATH AND PACKAGE_PATH)
  if (COMMON_INSTALL_PATH)
    set (INSTALL_PATH ${PACKAGE_PATH} CACHE PATH "Installation directory" FORCE)
  else ()
    set (INSTALL_PATH ${PACKAGE_PATH}/${PROJECT_NAME}-${PROJECT_VERSION}
      CACHE PATH "Installation directory" FORCE)
  endif ()
endif ()
if (INSTALL_PATH)
  file (TO_CMAKE_PATH ${INSTALL_PATH} CMAKE_INSTALL_PREFIX)
  message (STATUS "CMAKE_INSTALL_PREFIX set to ${CMAKE_INSTALL_PREFIX}")
endif ()

# (3) Build which libraries?  Possible values are SHARED, STATIC, BOTH.
if (MSVC)
  set (RANDOMLIB_LIB_TYPE STATIC CACHE STRING
    "Types of library generated: SHARED, STATIC (default), or BOTH")
else ()
  set (RANDOMLIB_LIB_TYPE SHARED CACHE STRING
    "Types of library generated: SHARED (default), STATIC, or BOTH")
endif ()
set_property (CACHE RANDOMLIB_LIB_TYPE
  PROPERTY STRINGS "SHARED" "STATIC" "BOTH")

if (RANDOMLIB_LIB_TYPE STREQUAL "BOTH")
  set (RANDOMLIB_SHARED_LIB ON)
  set (RANDOMLIB_STATIC_LIB ON)
  set (RANDOMLIB_LIB_TYPE_VAL 2)
elseif (RANDOMLIB_LIB_TYPE STREQUAL "SHARED")
  set (RANDOMLIB_SHARED_LIB ON)
  set (RANDOMLIB_STATIC_LIB OFF)
  set (RANDOMLIB_LIB_TYPE_VAL 1)
elseif (RANDOMLIB_LIB_TYPE STREQUAL "STATIC")
  set (RANDOMLIB_SHARED_LIB OFF)
  set (RANDOMLIB_STATIC_LIB ON)
  set (RANDOMLIB_LIB_TYPE_VAL 0)
else ()
  message (FATAL_ERROR
    "Bad value of RANDOMLIB_LIB_TYPE, \"${RANDOMLIB_LIB_TYPE}\" "
    "(should be SHARED, STATIC or BOTH)")
endif ()

if (RANDOMLIB_STATIC_LIB)
  set (PROJECT_STATIC_LIBRARIES RandomLib_STATIC)
  set (PROJECT_STATIC_DEFINITIONS -DRANDOMLIB_SHARED_LIB=0)
else ()
  set (PROJECT_STATIC_LIBRARIES)
  set (PROJECT_STATIC_DEFINITIONS)
endif ()

if (RANDOMLIB_SHARED_LIB)
  set (PROJECT_SHARED_LIBRARIES RandomLib)
  set (PROJECT_LIBRARIES ${PROJECT_SHARED_LIBRARIES})
  set (PROJECT_SHARED_DEFINITIONS -DRANDOMLIB_SHARED_LIB=1)
  set (PROJECT_DEFINITIONS ${PROJECT_SHARED_DEFINITIONS})
else ()
  set (PROJECT_SHARED_LIBRARIES)
  set (PROJECT_LIBRARIES ${PROJECT_STATIC_LIBRARIES})
  set (PROJECT_SHARED_DEFINITIONS)
  set (PROJECT_DEFINITIONS ${PROJECT_STATIC_DEFINITIONS})
endif ()

# (4) Create the documentation?  This depends on whether doxygen can be
# found.  If this is OFF, then links will be provided to the online
# documentation on Sourceforge.
option (RANDOMLIB_DOCUMENTATION "Use doxygen to create the documentation" OFF)

# (5) By default, cmake looks for hardware support of vector operations.
# This option allows you to disable this.
option (DISABLE_VECTOR_OPTIMIZATIONS "Do not look for SSE2 or AltiVec support"
  OFF)

# (6) By default, cmake looks for boost which is used by some of the
# examples. This option allows you to disable this.
option (DISABLE_BOOST "Do not look for boost libraries" OFF)

# (7) When making a binary package, should we include the debug version
# of the library?  This applies to MSVC only, because that's the
# platform where debug and release compilations do not inter-operate.
# It requires building as follows:
#   cmake --build . --config Debug   --target ALL_BUILD
#   cmake --build . --config Release --target ALL_BUILD
#   cmake --build . --config Release --target PACKAGE
option (PACKAGE_DEBUG_LIBS
  "Include debug versions of library in binary package" OFF)

# Finding other packages.  (These are only need for the examples.  The
# library does not depend on them.)

# On Windows we use boost's static libraries.
if (NOT DISABLE_BOOST)
  if (MSVC)
    set (Boost_USE_STATIC_LIBS ON)
  endif ()
  find_package (Boost QUIET COMPONENTS serialization date_time)
endif ()

# Optionally use OpenMP in RandomParallel.
#find_package (OpenMP QUIET)

set (LIBNAME Random)
if (MSVC OR CMAKE_CONFIGURATION_TYPES)
  # For multi-config systems and for Visual Studio, the debug version of
  # the library is called Random_d.
  set (CMAKE_DEBUG_POSTFIX _d)
endif ()

if (NOT MSVC)
  # Set the run time path for shared libraries for non-Windows machines.
  # (1) include link path for external packages (not needed with
  # RandomLib because there are no external packages).
  set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  # (2) include installed path for RandomLib
  if (NOT APPLE)
    # Use relative path so that package is relocatable
    set (CMAKE_INSTALL_RPATH "\$ORIGIN/../lib${LIB_SUFFIX}")
  else ()
    # Need absolute path with MacOSx
    set (CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}")
  endif ()
endif ()

# Look for vector support SSE2 for Intel chips and AltiVec for PowerPC.
include (CheckIncludeFileCXX)
if (DISABLE_VECTOR_OPTIMIZATIONS)
  set (VECTOR_FLAGS "")
  set (HAVE_SSE2 OFF)
  set (HAVE_ALTIVEC OFF)
else ()
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "i686" OR
      CMAKE_SYSTEM_PROCESSOR MATCHES "amd64" OR
      CMAKE_SYSTEM_PROCESSOR MATCHES "x86")
    check_include_file_cxx ("emmintrin.h" HAVE_SSE2)
    if (HAVE_SSE2)
      if (MSVC)
        set (VECTOR_FLAGS "/arch:SSE2")
      else ()
        set (VECTOR_FLAGS "-msse2")
      endif ()
    else ()
      set (VECTOR_FLAGS "")
    endif ()
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "powerpc")
    check_type_size ("vector unsigned" VECTOR_UNSIGNED BUILTIN_TYPES_ONLY)
    if (HAVE_VECTOR_UNSIGNED)
      set (VECTOR_FLAGS "-maltivec")
      set (HAVE_ALTIVEC ON)
    else ()
      set (VECTOR_FLAGS "")
      set (HAVE_ALTIVEC OFF)
    endif ()
  endif ()
endif ()

# Optionally compile MPFR example.  This requires MPFR 3.0 or later;
# the check for the version currently occurs in the MPFR source files.
check_include_file_cxx ("mpfr.h" HAVE_MPFR)
if (HAVE_MPFR)
  find_library (MPFR_LIBRARIES mpfr)
  find_library (GMP_LIBRARIES gmp)
  if (NOT (MPFR_LIBRARIES AND GMP_LIBRARIES))
    set (HAVE_MPFR OFF)
  endif ()
endif ()

# Determine system properties
include (CheckTypeSize)
check_type_size ("long double" LONG_DOUBLE BUILTIN_TYPES_ONLY)

# Create a Config.h to expose system information to the compiler
configure_file (
  include/RandomLib/Config.h.in
  include/RandomLib/Config.h )

# The documentation depends on doxygen.  Need version 1.8.1.2 or later
# because of use of &minus; etc.
if (RANDOMLIB_DOCUMENTATION)
  set (DOXYGEN_SKIP_DOT ON)
  find_package (Doxygen 1.8.1.2)
  if (DOXYGEN_FOUND)
    execute_process (COMMAND ${DOXYGEN_EXECUTABLE} --version
      OUTPUT_VARIABLE DOXYGEN_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (DOXYGEN_VERSION VERSION_LESS 1.8.1.2)
      set (DOXYGEN_FOUND FALSE)
      message (STATUS "Doxygen version found, ${DOXYGEN_VERSION}, is too old")
    endif ()
  endif ()
endif ()

# Set a default build type for single-configuration cmake generators if
# no build type is set.
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release)
endif ()

# Make the compiler more picky.  Also, on non-Windows machines, unroll
# loops in optimizing build types.
if (MSVC)
  string (REGEX REPLACE "/W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else ()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECTOR_FLAGS} -Wall -Wextra")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -funroll-loops")
  set (CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")
  # check for C++11 support
  include (CheckCXXCompilerFlag)
  set (CXX11FLAG "-std=c++11")
  check_cxx_compiler_flag (${CXX11FLAG} CXX11TEST1)
  if (NOT CXX11TEST1)
    set (CXX11FLAG "-std=c++0x")
    check_cxx_compiler_flag (${CXX11FLAG} CXX11TEST2)
    if (NOT CXX11TEST2)
      unset (CXX11FLAG)
    endif ()
  endif ()
  if (CXX11FLAG)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX11FLAG}")
  endif ()
endif ()

if (APPLE)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "i.86" OR
      CMAKE_SYSTEM_PROCESSOR MATCHES "amd64" OR
      CMAKE_SYSTEM_PROCESSOR MATCHES "x86")
    set (CMAKE_OSX_ARCHITECTURES "i386 -arch x86_64")
  endif ()
endif ()

# The list of examples to build.  Don't include MPFRExample here since
# it doesn't get linked with RandomLib.  This is treated specially in
# examples/CMakeLists.txt
set (EXAMPLES
  RandomExample RandomTime RandomThread RandomSave
  RandomExact RandomLambda RandomCoverage)

set_property (GLOBAL PROPERTY USE_FOLDERS ON)

# Set the include directories.  Look in ${PROJECT_BINARY_DIR}/include
# first because that's where Config.h will be
include_directories ("${PROJECT_BINARY_DIR}/include" include)

# The list of subdirectories to process
add_subdirectory (src)
add_subdirectory (include/RandomLib)
add_subdirectory (examples)
add_subdirectory (doc)
add_subdirectory (cmake)

# Packaging support; we deal with
# (1) a source distribution: cmake make a tar.gz file and the zip file
# is created from this.
# (2) a binary distribution: code is included for Linux, Apple, and
# Windows, but only the Windows distribution has been exercised.

# Need to ensure that system dlls get included in a binary distribution
if (NOT DEFINED CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_NO_WARNINGS)
  # Visual Studio Express does include redistributable components so
  # squelch the warning.
  set (CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_NO_WARNINGS ON)
endif ()
set (CMAKE_INSTALL_DEBUG_LIBRARIES ON)
include (InstallRequiredSystemLibraries)

# The configuration of CPack is via variables that need to be set before
# the include (CPack).
set (CPACK_PACKAGE_CONTACT charles@karney.com)
set (CPACK_PACKAGE_VENDOR "RandomLib")
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY
  "RandomLib library and documentation")
# The list of files to be excluded from the source distribution.
set (CPACK_SOURCE_IGNORE_FILES
  "#"
  "~\$"
  "/\\\\.git"
  "${PROJECT_SOURCE_DIR}/BUILD"
  "${PROJECT_SOURCE_DIR}/tests/"
  "${PROJECT_SOURCE_DIR}/distrib/"
  "${PROJECT_SOURCE_DIR}/[^/]*\\\\.html\$"
  "${PROJECT_SOURCE_DIR}/makefile-admin\$"
  "\\\\.eps\$" )
set (CPACK_SOURCE_GENERATOR TGZ)

set (CPACK_RESOURCE_FILE_LICENSE ${PROJECT_SOURCE_DIR}/LICENSE.txt)
set (CPACK_PACKAGE_INSTALL_DIRECTORY "${PROJECT_NAME}-${PROJECT_VERSION}")
set (CPACK_SOURCE_PACKAGE_FILE_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY}")

if (WIN32)
  # The Windows binary packager is NSIS.  Set the necessary variables
  # for this.
  set (CPACK_NSIS_CONTACT "charles@karney.com")
  set (CPACK_NSIS_URL_INFO_ABOUT "http://randomlib.sf.net")
  set (CPACK_NSIS_HELP_LINK "mailto:charles@karney.com")
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    # Hardcode the prefix for Visual Studio 10
    set (CPACK_NSIS_INSTALL_ROOT "C:\\\\pkg-vc10-x64")
    set (CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY}-win64")
    set (CPACK_NSIS_PACKAGE_NAME
      "${PROJECT_NAME} x64 ${PROJECT_VERSION}")
    set (CPACK_PACKAGE_INSTALL_REGISTRY_KEY
      "${PROJECT_NAME}-x64-${PROJECT_VERSION}")
  else ()
    # Hardcode the prefix for Visual Studio 10
    set (CPACK_NSIS_INSTALL_ROOT "C:\\\\pkg-vc10")
    set (CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY}-win32")
    set (CPACK_NSIS_PACKAGE_NAME
      "${PROJECT_NAME} ${PROJECT_VERSION}")
    set (CPACK_PACKAGE_INSTALL_REGISTRY_KEY
      "${PROJECT_NAME}-${PROJECT_VERSION}")
  endif ()
  set (CPACK_NSIS_DISPLAY_NAME ${CPACK_NSIS_PACKAGE_NAME})
  set (CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
  CreateShortCut \\\"$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\Library Documentation.lnk\\\" \\\"$INSTDIR\\\\share\\\\doc\\\\RandomLib\\\\html\\\\index.html\\\"
")
  set (CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "
  !insertmacro MUI_STARTMENU_GETFOLDER Application $MUI_TEMP
  Delete \\\"$SMPROGRAMS\\\\$MUI_TEMP\\\\Library Documentation.lnk\\\"
")
  set (CPACK_NSIS_MODIFY_PATH ON)
elseif (APPLE)
  # Not tested
  set (CPACK_GENERATOR DMG)
  set (CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY}-darwin")
else ()
  # Not tested
  set (CPACK_GENERATOR TGZ)
endif ()

include (CPack)

# Another maintainer-specific target is building the source distribution
# via the target dist.  This calls package_source to make a tar.gz file.
# However this needs to be touched up to support the vanilla Makefiles
# provided with RandomLib.  This entails
# (1) creating Makefile (which includes Makefile.mk);
# (2) creating a bare-bones Config.h (with just the version information);
# (3) making sure that make thinks the generated documentation files are
# up-to-date.
# Then a new tar.gz file and zip file are created.  To avoid potential
# problems with directory permissions, tar and zip are told only to
# archive the files.
if (NOT WIN32)
  add_custom_target (dist
    COMMAND ${CMAKE_MAKE_PROGRAM} package_source
    COMMAND
      cd _CPack_Packages/Linux-Source/TGZ/${CPACK_SOURCE_PACKAGE_FILE_NAME} &&
      echo include Makefile.mk > Makefile &&
      sed -e "s/Unconfigured/${PROJECT_VERSION}/"
      -e "s/MAJOR .*/MAJOR ${PROJECT_VERSION_MAJOR}/"
      -e "s/MINOR .*/MINOR ${PROJECT_VERSION_MINOR}/"
      -e "s/PATCH .*/PATCH ${PROJECT_VERSION_PATCH}/"
      include/RandomLib/Config.h > include/RandomLib/Config.h.new &&
      mv include/RandomLib/Config.h.new include/RandomLib/Config.h
    COMMAND
      cd _CPack_Packages/Linux-Source/TGZ/${CPACK_SOURCE_PACKAGE_FILE_NAME} &&
      chmod -R g-w .
    COMMAND
      cd _CPack_Packages/Linux-Source/TGZ &&
      find ${CPACK_SOURCE_PACKAGE_FILE_NAME} -type f |
      tar cfzT ${CMAKE_BINARY_DIR}/${CPACK_SOURCE_PACKAGE_FILE_NAME}.tar.gz -
    COMMAND
      rm -f ${CMAKE_BINARY_DIR}/${CPACK_SOURCE_PACKAGE_FILE_NAME}.zip &&
      rsync -a --delete
      _CPack_Packages/Linux-Source/TGZ/${CPACK_SOURCE_PACKAGE_FILE_NAME}
      _CPack_Packages/Linux-Source/TGZ.DOS/ &&
      cd _CPack_Packages/Linux-Source/TGZ.DOS &&
      find . -type f |
      egrep  '/\(doxyfile.in|MANIFEST.in|NEWS|AUTHORS|INSTALL|pom\\.xml|.*\\.\(cpp|hpp|h.in|txt|cmake.in|cmake|h\)\)$$' |
      xargs unix2dos -q -k &&
      find ${CPACK_SOURCE_PACKAGE_FILE_NAME} -type f |
      zip -q ${CMAKE_BINARY_DIR}/${CPACK_SOURCE_PACKAGE_FILE_NAME}.zip -@
  )
endif ()

# Add a test target; the tests are in tools.
enable_testing ()
