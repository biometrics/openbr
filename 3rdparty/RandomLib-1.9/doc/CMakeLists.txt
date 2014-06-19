if (COMMON_INSTALL_PATH)
  set (INSTALL_DOC_DIR "share/doc/RandomLib")
else ()
  set (INSTALL_DOC_DIR "doc")
endif ()

# Runs doxygen, if available

# First assemble a list of all the files the documentation uses.  Add a
# dependency on htmlman (from man/CMakeLists.txt).  Use html/index.html
# as the make target.  To make this target, copy the non-doxygen
# generated files into html/.  Run doxfile.in thru cmake's config
# process so that absolute path names are used and so that the pathnames
# are properly stripped by doxygen (via STRIP_FROM_PATH). The
# distrib-doc target copies the html directory into the source tree.

# If doxygen is not available, only the install step (from the source
# tree) is done.

if (DOXYGEN_FOUND)
  configure_file (doxyfile.in doxyfile)
  file (GLOB SOURCES
    ../src/[A-Za-z]*.cpp ../include/RandomLib/[A-Za-z]*.hpp
    ../examples/[A-Za-z]*.cpp)
  file (GLOB FIGURES *hist.png ky-5*.png *-bits.png)
  file (GLOB EXTRA_FILES *hist.pdf ../LICENSE.txt)
  file (COPY ${EXTRA_FILES} DESTINATION html-stage)
  add_custom_target (doc ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/index.html)
  add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doxyfile
      RandomLib.dox ${SOURCES} ${EXTRA_FILES} ${FIGURES}
    COMMAND ${CMAKE_COMMAND} -E remove_directory html
    COMMAND ${CMAKE_COMMAND} -E copy_directory html-stage html
    COMMAND ${DOXYGEN_EXECUTABLE} doxyfile > doxygen.log
    COMMENT "Generating html documentation tree")
  install (DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
    DESTINATION ${INSTALL_DOC_DIR})
else ()
  file (COPY ../LICENSE.txt DESTINATION html)
  configure_file (index.html.in html/index.html)
  install (FILES
    ${CMAKE_CURRENT_BINARY_DIR}/html/LICENSE.txt
    ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
    DESTINATION ${INSTALL_DOC_DIR}/html)
endif ()
