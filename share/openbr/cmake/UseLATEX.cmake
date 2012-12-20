# File: UseLATEX.cmake
# CMAKE commands to actually use the LaTeX compiler
# Version: 1.9.4
# Author: Kenneth Moreland <kmorel@sandia.gov>
#
# Copyright 2004 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
# license for use of this work by or on behalf of the
# U.S. Government. Redistribution and use in source and binary forms, with
# or without modification, are permitted provided that this Notice and any
# statement of authorship are reproduced on all copies.
#
# The following function is defined:
#
# ADD_LATEX_DOCUMENT(<tex_file>
#                    [BIBFILES <bib_files>]
#                    [INPUTS <input_tex_files>]
#                    [IMAGE_DIRS] <image_directories>
#                    [IMAGES] <image_files>
#                    [CONFIGURE] <tex_files>
#                    [DEPENDS] <tex_files>
#                    [MULTIBIB_NEWCITES] <suffix_list>
#                    [USE_INDEX] [USE_GLOSSARY] [USE_NOMENCL]
#                    [DEFAULT_PDF] [DEFAULT_SAFEPDF]
#                    [MANGLE_TARGET_NAMES])
#       Adds targets that compile <tex_file>.  The latex output is placed
#       in LATEX_OUTPUT_PATH or CMAKE_CURRENT_BINARY_DIR if the former is
#       not set.  The latex program is picky about where files are located,
#       so all input files are copied from the source directory to the
#       output directory.  This includes the target tex file, any tex file
#       listed with the INPUTS option, the bibliography files listed with
#       the BIBFILES option, and any .cls, .bst, and .clo files found in
#       the current source directory.  Images found in the IMAGE_DIRS
#       directories or listed by IMAGES are also copied to the output
#       directory and coverted to an appropriate format if necessary.  Any
#       tex files also listed with the CONFIGURE option are also processed
#       with the CMake CONFIGURE_FILE command (with the @ONLY flag).  Any
#       file listed in CONFIGURE but not the target tex file or listed with
#       INPUTS has no effect. DEPENDS can be used to specify generated files
#       that are needed to compile the latex target.
#
#       The following targets are made:
#               dvi: Makes <name>.dvi
#               pdf: Makes <name>.pdf using pdflatex.
#               safepdf: Makes <name>.pdf using ps2pdf.  If using the default
#                       program arguments, this will ensure all fonts are
#                       embedded and no lossy compression has been performed
#                       on images.
#               ps: Makes <name>.ps
#               html: Makes <name>.html
#               auxclean: Deletes <name>.aux and other auxiliary files.
#                       This is sometimes necessary if a LaTeX error occurs
#                       and writes a bad aux file.  Unlike the regular clean
#                       target, it does not delete other input files, such as
#                       converted images, to save time on the rebuild.
#
#       The dvi target is added to the ALL.  That is, it will be the target
#       built by default.  If the DEFAULT_PDF argument is given, then the
#       pdf target will be the default instead of dvi.
#
#       If the argument MANGLE_TARGET_NAMES is given, then each of the
#       target names above will be mangled with the <tex_file> name.  This
#       is to make the targets unique if ADD_LATEX_DOCUMENT is called for
#       multiple documents.  If the argument USE_INDEX is given, then
#       commands to build an index are made.  If the argument USE_GLOSSARY
#       is given, then commands to build a glossary are made.  If the
#       argument MULTIBIB_NEWCITES is given, then additional bibtex calls
#       are added to the build to support the extra auxiliary files created
#       with the \newcite command in the multibib package.
#
# History:
#
# 1.9.4 Fix issues with filenames containing multiple periods.
#
# 1.9.3 Hide some variables that are now cached but should not show up in
#       the ccmake list of variables.
#
# 1.9.2 Changed MACRO declarations to FUNCTION declarations.  The better
#       FUNCTION scoping will hopefully avoid some common but subtle bugs.
#       This implicitly increases the minimum CMake version to 4.6 (although
#       I honestly only test it with the latest 4.8 version).
#
#       Since we are updating the minimum CMake version, I'm going to start
#       using the builtin LIST commands that are now available.
#
#       Favor using pdftops from the Poppler package to convert from pdf to
#       eps.  It does a much better job than ImageMagick or ghostscript.
#
# 1.9.1 Fixed typo that caused the LATEX_SMALL_IMAGES option to fail to
#       activate.
#
# 1.9.0 Add support for the multibib package (thanks to Antonio LaTorre).
#
# 1.8.2 Fix corner case when an argument name was also a variable containing
#       the text of an argument.  In this case, the CMake IF was matching
#       the argument text with the contents of the variable with the same
#       argument name.
#
# 1.8.1 Fix problem where ps2pdf was not getting the appropriate arguments.
#
# 1.8.0 Add support for synctex.
#
# 1.7.7 Support calling xindy when making glossaries.
#
#       Improved make clean support.
#
# 1.7.6 Add support for the nomencl package (thanks to Myles English).
#
# 1.7.5 Fix issue with bibfiles being copied two different ways, which causes
#       Problems with dependencies (thanks to Edwin van Leeuwen).
#
# 1.7.4 Added the DEFAULT_SAFEPDF option (thanks to Raymond Wan).
#
#       Added warnings when image directories are not found (and were
#       probably not given relative to the source directory).
#
# 1.7.3 Fix some issues with interactions between makeglossaries and bibtex
#       (thanks to Mark de Wever).
#
# 1.7.2 Use ps2pdf to convert eps to pdf to get around the problem with
#       ImageMagick dropping the bounding box (thanks to Lukasz Lis).
#
# 1.7.1 Fixed some dependency issues.
#
# 1.7.0 Added DEPENDS options (thanks to Theodore Papadopoulo).
#
# 1.6.1 Ported the makeglossaries command to CMake and embedded the port
#       into UseLATEX.cmake.
#
# 1.6.0 Allow the use of the makeglossaries command.  Thanks to Oystein
#       S. Haaland for the patch.
#
# 1.5.0 Allow any type of file in the INPUTS lists, not just tex file
#       (suggested by Eric Noulard).  As a consequence, the ability to
#       specify tex files without the .tex extension is removed.  The removed
#       function is of dubious value anyway.
#
#       When copying input files, skip over any file that exists in the
#       binary directory but does not exist in the source directory with the
#       assumption that these files were added by some other mechanism.  I
#       find this useful when creating large documents with multiple
#       chapters that I want to build separately (for speed) as I work on
#       them.  I use the same boilerplate as the starting point for all
#       and just copy it with different configurations.  This was what the
#       separate ADD_LATEX_DOCUMENT method was supposed to originally be for.
#       Since its external use is pretty much deprecated, I removed that
#       documentation.
#
# 1.4.1 Copy .sty files along with the other class and package files.
#
# 1.4.0 Added a MANGLE_TARGET_NAMES option that will mangle the target names.
#
#       Fixed problem with copying bib files that became apparent with
#       CMake 2.4.
#
# 1.3.0 Added a LATEX_OUTPUT_PATH variable that allows you or the user to
#       specify where the built latex documents to go.  This is especially
#       handy if you want to do in-source builds.
#
#       Removed the ADD_LATEX_IMAGES macro and absorbed the functionality
#       into ADD_LATEX_DOCUMENT.  The old interface was always kind of
#       clunky anyway since you had to specify the image directory in both
#       places.  It also made supporting LATEX_OUTPUT_PATH problematic.
#
#       Added support for jpeg files.
#
# 1.2.0 Changed the configuration options yet again.  Removed the NO_CONFIGURE
#       Replaced it with a CONFIGURE option that lists input files for which
#       configure should be run.
#
#       The pdf target no longer depends on the dvi target.  This allows you
#       to build latex documents that require pdflatex.  Also added an option
#       to make the pdf target the default one.
#
# 1.1.1 Added the NO_CONFIGURE option.  The @ character can be used when
#       specifying table column separators.  If two or more are used, then
#       will incorrectly substitute them.
#
# 1.1.0 Added ability include multiple bib files.  Added ability to do copy
#       sub-tex files for multipart tex files.
#
# 1.0.0 If both ps and pdf type images exist, just copy the one that
#       matches the current render mode.  Replaced a bunch of STRING
#       commands with GET_FILENAME_COMPONENT commands that were made to do
#       the desired function.
#
# 0.4.0 First version posted to CMake Wiki.
#

#############################################################################
# Find the location of myself while originally executing.  If you do this
# inside of a macro, it will recode where the macro was invoked.
#############################################################################
SET(LATEX_USE_LATEX_LOCATION ${CMAKE_CURRENT_LIST_FILE}
  CACHE INTERNAL "Location of UseLATEX.cmake file." FORCE
  )

#############################################################################
# Generic helper functions
#############################################################################

FUNCTION(LATEX_LIST_CONTAINS var value)
  SET(input_list ${ARGN})
  LIST(FIND input_list "${value}" index)
  IF (index GREATER -1)
    SET(${var} TRUE PARENT_SCOPE)
  ELSE (index GREATER -1)
    SET(${var} PARENT_SCOPE)
  ENDIF (index GREATER -1)
ENDFUNCTION(LATEX_LIST_CONTAINS)

# Parse function arguments.  Variables containing the results are placed
# in the global scope for historical reasons.
FUNCTION(LATEX_PARSE_ARGUMENTS prefix arg_names option_names)
  SET(DEFAULT_ARGS)
  FOREACH(arg_name ${arg_names})
    SET(${prefix}_${arg_name} CACHE INTERNAL "${prefix} argument" FORCE)
  ENDFOREACH(arg_name)
  FOREACH(option ${option_names})
    SET(${prefix}_${option} CACHE INTERNAL "${prefix} option" FORCE)
  ENDFOREACH(option)

  SET(current_arg_name DEFAULT_ARGS)
  SET(current_arg_list)
  FOREACH(arg ${ARGN})
    LATEX_LIST_CONTAINS(is_arg_name ${arg} ${arg_names})
    LATEX_LIST_CONTAINS(is_option ${arg} ${option_names})
    IF (is_arg_name)
      SET(${prefix}_${current_arg_name} ${current_arg_list}
        CACHE INTERNAL "${prefix} argument" FORCE)
      SET(current_arg_name ${arg})
      SET(current_arg_list)
    ELSEIF (is_option)
      SET(${prefix}_${arg} TRUE CACHE INTERNAL "${prefix} option" FORCE)
    ELSE (is_arg_name)
      SET(current_arg_list ${current_arg_list} ${arg})
    ENDIF (is_arg_name)
  ENDFOREACH(arg)
  SET(${prefix}_${current_arg_name} ${current_arg_list}
    CACHE INTERNAL "${prefix} argument" FORCE)
ENDFUNCTION(LATEX_PARSE_ARGUMENTS)

# Match the contents of a file to a regular expression.
FUNCTION(LATEX_FILE_MATCH variable filename regexp default)
  # The FILE STRINGS command would be a bit better, but I'm not totally sure
  # the match will always be to a whole line, and I don't want to break things.
  FILE(READ ${filename} file_contents)
  STRING(REGEX MATCHALL "${regexp}"
    match_result ${file_contents}
    )
  IF (match_result)
    SET(${variable} "${match_result}" PARENT_SCOPE)
  ELSE (match_result)
    SET(${variable} "${default}" PARENT_SCOPE)
  ENDIF (match_result)
ENDFUNCTION(LATEX_FILE_MATCH)

# A version of GET_FILENAME_COMPONENT that treats extensions after the last
# period rather than the first.  To the best of my knowledge, all filenames
# typically used by LaTeX, including image files, have small extensions
# after the last dot.
FUNCTION(LATEX_GET_FILENAME_COMPONENT varname filename type)
  SET(result)
  IF ("${type}" STREQUAL "NAME_WE")
    GET_FILENAME_COMPONENT(name ${filename} NAME)
    STRING(REGEX REPLACE "\\.[^.]*\$" "" result "${name}")
  ELSEIF ("${type}" STREQUAL "EXT")
    GET_FILENAME_COMPONENT(name ${filename} NAME)
    STRING(REGEX MATCH "\\.[^.]*\$" result "${name}")
  ELSE ("${type}" STREQUAL "NAME_WE")
    GET_FILENAME_COMPONENT(result ${filename} ${type})
  ENDIF ("${type}" STREQUAL "NAME_WE")
  SET(${varname} "${result}" PARENT_SCOPE)
ENDFUNCTION(LATEX_GET_FILENAME_COMPONENT)

#############################################################################
# Functions that perform processing during a LaTeX build.
#############################################################################
FUNCTION(LATEX_MAKEGLOSSARIES)
  # This is really a bare bones port of the makeglossaries perl script into
  # CMake scripting.
  MESSAGE("**************************** In makeglossaries")
  IF (NOT LATEX_TARGET)
    MESSAGE(SEND_ERROR "Need to define LATEX_TARGET")
  ENDIF (NOT LATEX_TARGET)

  SET(aux_file ${LATEX_TARGET}.aux)

  IF (NOT EXISTS ${aux_file})
    MESSAGE(SEND_ERROR "${aux_file} does not exist.  Run latex on your target file.")
  ENDIF (NOT EXISTS ${aux_file})

  LATEX_FILE_MATCH(newglossary_lines ${aux_file}
    "@newglossary[ \t]*{([^}]*)}{([^}]*)}{([^}]*)}{([^}]*)}"
    "@newglossary{main}{glg}{gls}{glo}"
    )

  LATEX_FILE_MATCH(istfile_line ${aux_file}
    "@istfilename[ \t]*{([^}]*)}"
    "@istfilename{${LATEX_TARGET}.ist}"
    )
  STRING(REGEX REPLACE "@istfilename[ \t]*{([^}]*)}" "\\1"
    istfile ${istfile_line}
    )

  STRING(REGEX MATCH ".*\\.xdy" use_xindy "${istfile}")
  IF (use_xindy)
    MESSAGE("*************** Using xindy")
    IF (NOT XINDY_COMPILER)
      MESSAGE(SEND_ERROR "Need to define XINDY_COMPILER")
    ENDIF (NOT XINDY_COMPILER)
  ELSE (use_xindy)
    MESSAGE("*************** Using makeindex")
    IF (NOT MAKEINDEX_COMPILER)
      MESSAGE(SEND_ERROR "Need to define MAKEINDEX_COMPILER")
    ENDIF (NOT MAKEINDEX_COMPILER)
  ENDIF (use_xindy)

  FOREACH(newglossary ${newglossary_lines})
    STRING(REGEX REPLACE
      "@newglossary[ \t]*{([^}]*)}{([^}]*)}{([^}]*)}{([^}]*)}"
      "\\1" glossary_name ${newglossary}
      )
    STRING(REGEX REPLACE
      "@newglossary[ \t]*{([^}]*)}{([^}]*)}{([^}]*)}{([^}]*)}"
      "${LATEX_TARGET}.\\2" glossary_log ${newglossary}
      )
    STRING(REGEX REPLACE
      "@newglossary[ \t]*{([^}]*)}{([^}]*)}{([^}]*)}{([^}]*)}"
      "${LATEX_TARGET}.\\3" glossary_out ${newglossary}
      )
    STRING(REGEX REPLACE
      "@newglossary[ \t]*{([^}]*)}{([^}]*)}{([^}]*)}{([^}]*)}"
      "${LATEX_TARGET}.\\4" glossary_in ${newglossary}
      )

    IF (use_xindy)
      LATEX_FILE_MATCH(xdylanguage_line ${aux_file}
        "@xdylanguage[ \t]*{${glossary_name}}{([^}]*)}"
        "@xdylanguage{${glossary_name}}{english}"
        )
      STRING(REGEX REPLACE
        "@xdylanguage[ \t]*{${glossary_name}}{([^}]*)}"
        "\\1"
        language
        ${xdylanguage_line}
        )
      # What crazy person makes a LaTeX index generater that uses different
      # identifiers for language than babel (or at least does not support
      # the old ones)?
      IF (${language} STREQUAL "frenchb")
        SET(language "french")
      ELSEIF (${language} MATCHES "^n?germanb?$")
        SET(language "german")
      ELSEIF (${language} STREQUAL "magyar")
        SET(language "hungarian")
      ELSEIF (${language} STREQUAL "lsorbian")
        SET(language "lower-sorbian")
      ELSEIF (${language} STREQUAL "norsk")
        SET(language "norwegian")
      ELSEIF (${language} STREQUAL "portuges")
        SET(language "portuguese")
      ELSEIF (${language} STREQUAL "russianb")
        SET(language "russian")
      ELSEIF (${language} STREQUAL "slovene")
        SET(language "slovenian")
      ELSEIF (${language} STREQUAL "ukraineb")
        SET(language "ukrainian")
      ELSEIF (${language} STREQUAL "usorbian")
        SET(language "upper-sorbian")
      ENDIF (${language} STREQUAL "frenchb")
      IF (language)
        SET(language_flags "-L ${language}")
      ELSE (language)
        SET(language_flags "")
      ENDIF (language)

      LATEX_FILE_MATCH(codepage_line ${aux_file}
        "@gls@codepage[ \t]*{${glossary_name}}{([^}]*)}"
        "@gls@codepage{${glossary_name}}{utf}"
        )
      STRING(REGEX REPLACE
        "@gls@codepage[ \t]*{${glossary_name}}{([^}]*)}"
        "\\1"
        codepage
        ${codepage_line}
        )
      IF (codepage)
        SET(codepage_flags "-C ${codepage}")
      ELSE (codepage)
        # Ideally, we would check that the language is compatible with the
        # default codepage, but I'm hoping that distributions will be smart
        # enough to specify their own codepage.  I know, it's asking a lot.
        SET(codepage_flags "")
      ENDIF (codepage)

      MESSAGE("${XINDY_COMPILER} ${MAKEGLOSSARIES_COMPILER_FLAGS} ${language_flags} ${codepage_flags} -I xindy -M ${glossary_name} -t ${glossary_log} -o ${glossary_out} ${glossary_in}"
        )
      EXEC_PROGRAM(${XINDY_COMPILER}
        ARGS ${MAKEGLOSSARIES_COMPILER_FLAGS}
          ${language_flags}
          ${codepage_flags}
          -I xindy
          -M ${glossary_name}
          -t ${glossary_log}
          -o ${glossary_out}
          ${glossary_in}
        OUTPUT_VARIABLE xindy_output
        )
      MESSAGE("${xindy_output}")

      # So, it is possible (perhaps common?) for aux files to specify a
      # language and codepage that are incompatible with each other.  Check
      # for that condition, and if it happens run again with the default
      # codepage.
      IF ("${xindy_output}" MATCHES "^Cannot locate xindy module for language (.+) in codepage (.+)\\.$")
        MESSAGE("*************** Retrying xindy with default codepage.")
        EXEC_PROGRAM(${XINDY_COMPILER}
          ARGS ${MAKEGLOSSARIES_COMPILER_FLAGS}
            ${language_flags}
            -I xindy
            -M ${glossary_name}
            -t ${glossary_log}
            -o ${glossary_out}
            ${glossary_in}
          )
      ENDIF ("${xindy_output}" MATCHES "^Cannot locate xindy module for language (.+) in codepage (.+)\\.$")
      #ENDIF ("${xindy_output}" MATCHES "Cannot locate xindy module for language (.+) in codepage (.+)\\.")
      
    ELSE (use_xindy)
      MESSAGE("${MAKEINDEX_COMPILER} ${MAKEGLOSSARIES_COMPILER_FLAGS} -s ${istfile} -t ${glossary_log} -o ${glossary_out} ${glossary_in}")
      EXEC_PROGRAM(${MAKEINDEX_COMPILER} ARGS ${MAKEGLOSSARIES_COMPILER_FLAGS}
        -s ${istfile} -t ${glossary_log} -o ${glossary_out} ${glossary_in}
        )
    ENDIF (use_xindy)

  ENDFOREACH(newglossary)
ENDFUNCTION(LATEX_MAKEGLOSSARIES)

FUNCTION(LATEX_MAKENOMENCLATURE)
  MESSAGE("**************************** In makenomenclature")
  IF (NOT LATEX_TARGET)
    MESSAGE(SEND_ERROR "Need to define LATEX_TARGET")
  ENDIF (NOT LATEX_TARGET)

  IF (NOT MAKEINDEX_COMPILER)
    MESSAGE(SEND_ERROR "Need to define MAKEINDEX_COMPILER")
  ENDIF (NOT MAKEINDEX_COMPILER)

  SET(nomencl_out ${LATEX_TARGET}.nls)
  SET(nomencl_in ${LATEX_TARGET}.nlo)

  EXEC_PROGRAM(${MAKEINDEX_COMPILER} ARGS ${MAKENOMENCLATURE_COMPILER_FLAGS}
    ${nomencl_in} -s "nomencl.ist" -o ${nomencl_out}
    )
ENDFUNCTION(LATEX_MAKENOMENCLATURE)

FUNCTION(LATEX_CORRECT_SYNCTEX)
  MESSAGE("**************************** In correct SyncTeX")
  IF (NOT LATEX_TARGET)
    MESSAGE(SEND_ERROR "Need to define LATEX_TARGET")
  ENDIF (NOT LATEX_TARGET)

  IF (NOT GZIP)
    MESSAGE(SEND_ERROR "Need to define GZIP")
  ENDIF (NOT GZIP)

  IF (NOT LATEX_SOURCE_DIRECTORY)
    MESSAGE(SEND_ERROR "Need to define LATEX_SOURCE_DIRECTORY")
  ENDIF (NOT LATEX_SOURCE_DIRECTORY)

  IF (NOT LATEX_BINARY_DIRECTORY)
    MESSAGE(SEND_ERROR "Need to define LATEX_BINARY_DIRECTORY")
  ENDIF (NOT LATEX_BINARY_DIRECTORY)

  SET(synctex_file ${LATEX_BINARY_DIRECTORY}/${LATEX_TARGET}.synctex)
  SET(synctex_file_gz ${synctex_file}.gz)

  IF (EXISTS ${synctex_file_gz})

    MESSAGE("Making backup of synctex file.")
    CONFIGURE_FILE(${synctex_file_gz} ${synctex_file}.bak.gz COPYONLY)

    MESSAGE("Uncompressing synctex file.")
    EXEC_PROGRAM(${GZIP}
      ARGS --decompress ${synctex_file_gz}
      )

    MESSAGE("Reading synctex file.")
    FILE(READ ${synctex_file} synctex_data)

    MESSAGE("Replacing relative with absolute paths.")
    STRING(REGEX REPLACE
      "(Input:[0-9]+:)([^/\n][^\n]*)"
      "\\1${LATEX_SOURCE_DIRECTORY}/\\2"
      synctex_data
      "${synctex_data}"
      )

    MESSAGE("Writing synctex file.")
    FILE(WRITE ${synctex_file} "${synctex_data}")

    MESSAGE("Compressing synctex file.")
    EXEC_PROGRAM(${GZIP}
      ARGS ${synctex_file}
      )

  ELSE (EXISTS ${synctex_file_gz})

    MESSAGE(SEND_ERROR "File ${synctex_file_gz} not found.  Perhaps synctex is not supported by your LaTeX compiler.")

  ENDIF (EXISTS ${synctex_file_gz})

ENDFUNCTION(LATEX_CORRECT_SYNCTEX)

#############################################################################
# Helper functions for establishing LaTeX build.
#############################################################################

FUNCTION(LATEX_NEEDIT VAR NAME)
  IF (NOT ${VAR})
    MESSAGE(SEND_ERROR "I need the ${NAME} command.")
  ENDIF(NOT ${VAR})
ENDFUNCTION(LATEX_NEEDIT)

FUNCTION(LATEX_WANTIT VAR NAME)
  IF (NOT ${VAR})
    MESSAGE(STATUS "I could not find the ${NAME} command.")
  ENDIF(NOT ${VAR})
ENDFUNCTION(LATEX_WANTIT)

FUNCTION(LATEX_SETUP_VARIABLES)
  SET(LATEX_OUTPUT_PATH "${LATEX_OUTPUT_PATH}"
    CACHE PATH "If non empty, specifies the location to place LaTeX output."
    )

  FIND_PACKAGE(LATEX)

  FIND_PROGRAM(XINDY_COMPILER
    NAME xindy
    PATHS ${MIKTEX_BINARY_PATH} /usr/bin
    )

  FIND_PACKAGE(UnixCommands)

  FIND_PROGRAM(PDFTOPS_CONVERTER
    NAMES pdftops
    DOC "The pdf to ps converter program from the Poppler package."
    )

  MARK_AS_ADVANCED(CLEAR
    LATEX_COMPILER
    PDFLATEX_COMPILER
    BIBTEX_COMPILER
    MAKEINDEX_COMPILER
    XINDY_COMPILER
    DVIPS_CONVERTER
    PS2PDF_CONVERTER
    PDFTOPS_CONVERTER
    LATEX2HTML_CONVERTER
    )

  LATEX_NEEDIT(LATEX_COMPILER latex)
  LATEX_WANTIT(PDFLATEX_COMPILER pdflatex)
  LATEX_NEEDIT(BIBTEX_COMPILER bibtex)
  LATEX_NEEDIT(MAKEINDEX_COMPILER makeindex)
  LATEX_WANTIT(DVIPS_CONVERTER dvips)
  LATEX_WANTIT(PS2PDF_CONVERTER ps2pdf)
  LATEX_WANTIT(PDFTOPS_CONVERTER pdftops)
  LATEX_WANTIT(LATEX2HTML_CONVERTER latex2html)

  SET(LATEX_COMPILER_FLAGS "-interaction=nonstopmode"
    CACHE STRING "Flags passed to latex.")
  SET(PDFLATEX_COMPILER_FLAGS ${LATEX_COMPILER_FLAGS}
    CACHE STRING "Flags passed to pdflatex.")
  SET(LATEX_SYNCTEX_FLAGS "-synctex=1"
    CACHE STRING "latex/pdflatex flags used to create synctex file.")
  SET(BIBTEX_COMPILER_FLAGS ""
    CACHE STRING "Flags passed to bibtex.")
  SET(MAKEINDEX_COMPILER_FLAGS ""
    CACHE STRING "Flags passed to makeindex.")
  SET(MAKEGLOSSARIES_COMPILER_FLAGS ""
    CACHE STRING "Flags passed to makeglossaries.")
  SET(MAKENOMENCLATURE_COMPILER_FLAGS ""
    CACHE STRING "Flags passed to makenomenclature.")
  SET(DVIPS_CONVERTER_FLAGS "-Ppdf -G0 -t letter"
    CACHE STRING "Flags passed to dvips.")
  SET(PS2PDF_CONVERTER_FLAGS "-dMaxSubsetPct=100 -dCompatibilityLevel=1.3 -dSubsetFonts=true -dEmbedAllFonts=true -dAutoFilterColorImages=false -dAutoFilterGrayImages=false -dColorImageFilter=/FlateEncode -dGrayImageFilter=/FlateEncode -dMonoImageFilter=/FlateEncode"
    CACHE STRING "Flags passed to ps2pdf.")
  SET(PDFTOPS_CONVERTER_FLAGS -r 600
    CACHE STRING "Flags passed to pdftops.")
  SET(LATEX2HTML_CONVERTER_FLAGS ""
    CACHE STRING "Flags passed to latex2html.")
  MARK_AS_ADVANCED(
    LATEX_COMPILER_FLAGS
    PDFLATEX_COMPILER_FLAGS
    LATEX_SYNCTEX_FLAGS
    BIBTEX_COMPILER_FLAGS
    MAKEINDEX_COMPILER_FLAGS
    MAKEGLOSSARIES_COMPILER_FLAGS
    MAKENOMENCLATURE_COMPILER_FLAGS
    DVIPS_CONVERTER_FLAGS
    PS2PDF_CONVERTER_FLAGS
    PDFTOPS_CONVERTER_FLAGS
    LATEX2HTML_CONVERTER_FLAGS
    )
  SEPARATE_ARGUMENTS(LATEX_COMPILER_FLAGS)
  SEPARATE_ARGUMENTS(PDFLATEX_COMPILER_FLAGS)
  SEPARATE_ARGUMENTS(LATEX_SYNCTEX_FLAGS)
  SEPARATE_ARGUMENTS(BIBTEX_COMPILER_FLAGS)
  SEPARATE_ARGUMENTS(MAKEINDEX_COMPILER_FLAGS)
  SEPARATE_ARGUMENTS(MAKEGLOSSARIES_COMPILER_FLAGS)
  SEPARATE_ARGUMENTS(MAKENOMENCLATURE_COMPILER_FLAGS)
  SEPARATE_ARGUMENTS(DVIPS_CONVERTER_FLAGS)
  SEPARATE_ARGUMENTS(PS2PDF_CONVERTER_FLAGS)
  SEPARATE_ARGUMENTS(PDFTOPS_CONVERTER_FLAGS)
  SEPARATE_ARGUMENTS(LATEX2HTML_CONVERTER_FLAGS)

  FIND_PROGRAM(IMAGEMAGICK_CONVERT convert
    DOC "The convert program that comes with ImageMagick (available at http://www.imagemagick.org)."
    )
  IF (NOT IMAGEMAGICK_CONVERT)
    MESSAGE(SEND_ERROR "Could not find convert program.  Please download ImageMagick from http://www.imagemagick.org and install.")
  ENDIF (NOT IMAGEMAGICK_CONVERT)

  OPTION(LATEX_USE_SYNCTEX
    "If on, have LaTeX generate a synctex file, which WYSIWYG editors can use to correlate output files like dvi and pdf with the lines of LaTeX source that generates them.  In addition to adding the LATEX_SYNCTEX_FLAGS to the command line, this option also adds build commands that \"corrects\" the resulting synctex file to point to the original LaTeX files rather than those generated by UseLATEX.cmake."
    OFF
    )

  OPTION(LATEX_SMALL_IMAGES
    "If on, the raster images will be converted to 1/6 the original size.  This is because papers usually require 600 dpi images whereas most monitors only require at most 96 dpi.  Thus, smaller images make smaller files for web distributation and can make it faster to read dvi files."
    OFF)
  IF (LATEX_SMALL_IMAGES)
    SET(LATEX_RASTER_SCALE 16)
    SET(LATEX_OPPOSITE_RASTER_SCALE 100)
  ELSE (LATEX_SMALL_IMAGES)
    SET(LATEX_RASTER_SCALE 100)
    SET(LATEX_OPPOSITE_RASTER_SCALE 16)
  ENDIF (LATEX_SMALL_IMAGES)

  # Just holds extensions for known image types.  They should all be lower case.
  # For historical reasons, these are all declared in the global scope.
  SET(LATEX_DVI_VECTOR_IMAGE_EXTENSIONS .eps CACHE INTERNAL "")
  SET(LATEX_DVI_RASTER_IMAGE_EXTENSIONS CACHE INTERNAL "")
  SET(LATEX_DVI_IMAGE_EXTENSIONS
    ${LATEX_DVI_VECTOR_IMAGE_EXTENSIONS} ${LATEX_DVI_RASTER_IMAGE_EXTENSIONS}
    CACHE INTERNAL "")
  SET(LATEX_PDF_VECTOR_IMAGE_EXTENSIONS .pdf CACHE INTERNAL "")
  SET(LATEX_PDF_RASTER_IMAGE_EXTENSIONS .png .jpeg .jpg CACHE INTERNAL "")
  SET(LATEX_PDF_IMAGE_EXTENSIONS
    ${LATEX_PDF_VECTOR_IMAGE_EXTENSIONS} ${LATEX_PDF_RASTER_IMAGE_EXTENSIONS}
    CACHE INTERNAL "")
  SET(LATEX_IMAGE_EXTENSIONS
    ${LATEX_DVI_IMAGE_EXTENSIONS} ${LATEX_PDF_IMAGE_EXTENSIONS}
    CACHE INTERNAL "")
ENDFUNCTION(LATEX_SETUP_VARIABLES)

FUNCTION(LATEX_GET_OUTPUT_PATH var)
  SET(latex_output_path)
  IF (LATEX_OUTPUT_PATH)
    IF ("${LATEX_OUTPUT_PATH}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
      MESSAGE(SEND_ERROR "You cannot set LATEX_OUTPUT_PATH to the same directory that contains LaTeX input files.")
    ELSE ("${LATEX_OUTPUT_PATH}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
      SET(latex_output_path "${LATEX_OUTPUT_PATH}")
    ENDIF ("${LATEX_OUTPUT_PATH}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  ELSE (LATEX_OUTPUT_PATH)
    IF ("${CMAKE_CURRENT_BINARY_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
      MESSAGE(SEND_ERROR "LaTeX files must be built out of source or you must set LATEX_OUTPUT_PATH.")
    ELSE ("${CMAKE_CURRENT_BINARY_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
      SET(latex_output_path "${CMAKE_CURRENT_BINARY_DIR}")
    ENDIF ("${CMAKE_CURRENT_BINARY_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  ENDIF (LATEX_OUTPUT_PATH)
  SET(${var} ${latex_output_path} PARENT_SCOPE)
ENDFUNCTION(LATEX_GET_OUTPUT_PATH)

FUNCTION(LATEX_ADD_CONVERT_COMMAND
    output_path
    input_path
    output_extension
    input_extension
    flags
    )
  SET (converter ${IMAGEMAGICK_CONVERT})
  SET (convert_flags "")
  IF (${input_extension} STREQUAL ".eps" AND ${output_extension} STREQUAL ".pdf")
    # ImageMagick has broken eps to pdf conversion
    # use ps2pdf instead
    IF (PS2PDF_CONVERTER)
      SET (converter ${PS2PDF_CONVERTER})
      SET (convert_flags -dEPSCrop ${PS2PDF_CONVERTER_FLAGS})
    ELSE (PS2PDF_CONVERTER)
      MESSAGE(SEND_ERROR "Using postscript files with pdflatex requires ps2pdf for conversion.")
    ENDIF (PS2PDF_CONVERTER)
  ELSEIF (${input_extension} STREQUAL ".pdf" AND ${output_extension} STREQUAL ".eps")
    # ImageMagick can also be sketchy on pdf to eps conversion.  Not good with
    # color spaces and tends to unnecessarily rasterize.
    # use pdftops instead
    IF (PDFTOPS_CONVERTER)
      SET(converter ${PDFTOPS_CONVERTER})
      SET(convert_flags -eps ${PDFTOPS_CONVERTER_FLAGS})
    ELSE (PDFTOPS_CONVERTER)
      MESSAGE(STATUS "Consider getting pdftops from Poppler to convert PDF images to EPS images.")
      SET (convert_flags ${flags})
    ENDIF (PDFTOPS_CONVERTER)
  ELSE (${input_extension} STREQUAL ".eps" AND ${output_extension} STREQUAL ".pdf")
    SET (convert_flags ${flags})
  ENDIF (${input_extension} STREQUAL ".eps" AND ${output_extension} STREQUAL ".pdf")

  ADD_CUSTOM_COMMAND(OUTPUT ${output_path}
    COMMAND ${converter}
      ARGS ${convert_flags} ${input_path} ${output_path}
    DEPENDS ${input_path}
    )
ENDFUNCTION(LATEX_ADD_CONVERT_COMMAND)

# Makes custom commands to convert a file to a particular type.
FUNCTION(LATEX_CONVERT_IMAGE
    output_files_var
    input_file
    output_extension
    convert_flags
    output_extensions
    other_files
    )
  SET(output_file_list)
  SET(input_dir ${CMAKE_CURRENT_SOURCE_DIR})
  LATEX_GET_OUTPUT_PATH(output_dir)

  LATEX_GET_FILENAME_COMPONENT(extension "${input_file}" EXT)

  # Check input filename for potential problems with LaTeX.
  LATEX_GET_FILENAME_COMPONENT(name "${input_file}" NAME_WE)
  IF (name MATCHES ".*\\..*")
    STRING(REPLACE "." "-" suggested_name "${name}")
    SET(suggested_name "${suggested_name}${extension}")
    MESSAGE(WARNING "Some LaTeX distributions have problems with image file names with multiple extensions.  Consider changing ${name}${extension} to something like ${suggested_name}.")
  ENDIF (name MATCHES ".*\\..*")

  STRING(REGEX REPLACE "\\.[^.]*\$" ${output_extension} output_file
    "${input_file}")

  LATEX_LIST_CONTAINS(is_type ${extension} ${output_extensions})
  IF (is_type)
    IF (convert_flags)
      LATEX_ADD_CONVERT_COMMAND(${output_dir}/${output_file}
        ${input_dir}/${input_file} ${output_extension} ${extension}
        "${convert_flags}")
      SET(output_file_list ${output_file_list} ${output_dir}/${output_file})
    ELSE (convert_flags)
      # As a shortcut, we can just copy the file.
      ADD_CUSTOM_COMMAND(OUTPUT ${output_dir}/${input_file}
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${input_dir}/${input_file} ${output_dir}/${input_file}
        DEPENDS ${input_dir}/${input_file}
        )
      SET(output_file_list ${output_file_list} ${output_dir}/${input_file})
    ENDIF (convert_flags)
  ELSE (is_type)
    SET(do_convert TRUE)
    # Check to see if there is another input file of the appropriate type.
    FOREACH(valid_extension ${output_extensions})
      STRING(REGEX REPLACE "\\.[^.]*\$" ${output_extension} try_file
        "${input_file}")
      LATEX_LIST_CONTAINS(has_native_file "${try_file}" ${other_files})
      IF (has_native_file)
        SET(do_convert FALSE)
      ENDIF (has_native_file)
    ENDFOREACH(valid_extension)

    # If we still need to convert, do it.
    IF (do_convert)
      LATEX_ADD_CONVERT_COMMAND(${output_dir}/${output_file}
        ${input_dir}/${input_file} ${output_extension} ${extension}
        "${convert_flags}")
      SET(output_file_list ${output_file_list} ${output_dir}/${output_file})
    ENDIF (do_convert)
  ENDIF (is_type)

  SET(${output_files_var} ${output_file_list} PARENT_SCOPE)
ENDFUNCTION(LATEX_CONVERT_IMAGE)

# Adds custom commands to process the given files for dvi and pdf builds.
# Adds the output files to the given variables (does not replace).
FUNCTION(LATEX_PROCESS_IMAGES dvi_outputs_var pdf_outputs_var)
  LATEX_GET_OUTPUT_PATH(output_dir)
  SET(dvi_outputs)
  SET(pdf_outputs)
  FOREACH(file ${ARGN})
    IF (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
      LATEX_GET_FILENAME_COMPONENT(extension "${file}" EXT)
      SET(convert_flags)

      # Check to see if we need to downsample the image.
      LATEX_LIST_CONTAINS(is_raster "${extension}"
        ${LATEX_DVI_RASTER_IMAGE_EXTENSIONS}
        ${LATEX_PDF_RASTER_IMAGE_EXTENSIONS})
      IF (LATEX_SMALL_IMAGES)
        IF (is_raster)
          SET(convert_flags -resize ${LATEX_RASTER_SCALE}%)
        ENDIF (is_raster)
      ENDIF (LATEX_SMALL_IMAGES)

      # Make sure the output directory exists.
      LATEX_GET_FILENAME_COMPONENT(path "${output_dir}/${file}" PATH)
      MAKE_DIRECTORY("${path}")

      # Do conversions for dvi.
      LATEX_CONVERT_IMAGE(output_files "${file}" .eps "${convert_flags}"
        "${LATEX_DVI_IMAGE_EXTENSIONS}" "${ARGN}")
      SET(dvi_outputs ${dvi_outputs} ${output_files})

      # Do conversions for pdf.
      IF (is_raster)
        LATEX_CONVERT_IMAGE(output_files "${file}" .png "${convert_flags}"
          "${LATEX_PDF_IMAGE_EXTENSIONS}" "${ARGN}")
        SET(pdf_outputs ${pdf_outputs} ${output_files})
      ELSE (is_raster)
        LATEX_CONVERT_IMAGE(output_files "${file}" .pdf "${convert_flags}"
          "${LATEX_PDF_IMAGE_EXTENSIONS}" "${ARGN}")
        SET(pdf_outputs ${pdf_outputs} ${output_files})
      ENDIF (is_raster)
    ELSE (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
      MESSAGE(WARNING "Could not find file ${CMAKE_CURRENT_SOURCE_DIR}/${file}.  Are you sure you gave relative paths to IMAGES?")
    ENDIF (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
  ENDFOREACH(file)

  SET(${dvi_outputs_var} ${dvi_outputs} PARENT_SCOPE)
  SET(${pdf_outputs_var} ${pdf_outputs} PARENT_SCOPE)
ENDFUNCTION(LATEX_PROCESS_IMAGES)

FUNCTION(ADD_LATEX_IMAGES)
  MESSAGE(SEND_ERROR "The ADD_LATEX_IMAGES function is deprecated.  Image directories are specified with LATEX_ADD_DOCUMENT.")
ENDFUNCTION(ADD_LATEX_IMAGES)

FUNCTION(LATEX_COPY_GLOBBED_FILES pattern dest)
  FILE(GLOB file_list ${pattern})
  FOREACH(in_file ${file_list})
    LATEX_GET_FILENAME_COMPONENT(out_file ${in_file} NAME)
    CONFIGURE_FILE(${in_file} ${dest}/${out_file} COPYONLY)
  ENDFOREACH(in_file)
ENDFUNCTION(LATEX_COPY_GLOBBED_FILES)

FUNCTION(LATEX_COPY_INPUT_FILE file)
  LATEX_GET_OUTPUT_PATH(output_dir)

  IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    LATEX_GET_FILENAME_COMPONENT(path ${file} PATH)
    FILE(MAKE_DIRECTORY ${output_dir}/${path})

    LATEX_LIST_CONTAINS(use_config ${file} ${LATEX_CONFIGURE})
    IF (use_config)
      CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${file}
        ${output_dir}/${file}
        @ONLY
        )
      ADD_CUSTOM_COMMAND(OUTPUT ${output_dir}/${file}
        COMMAND ${CMAKE_COMMAND}
        ARGS ${CMAKE_BINARY_DIR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file}
        )
    ELSE (use_config)
      ADD_CUSTOM_COMMAND(OUTPUT ${output_dir}/${file}
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${output_dir}/${file}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file}
        )
    ENDIF (use_config)
  ELSE (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    IF (EXISTS ${output_dir}/${file})
      # Special case: output exists but input does not.  Assume that it was
      # created elsewhere and skip the input file copy.
    ELSE (EXISTS ${output_dir}/${file})
      MESSAGE("Could not find input file ${CMAKE_CURRENT_SOURCE_DIR}/${file}")
    ENDIF (EXISTS ${output_dir}/${file})
  ENDIF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
ENDFUNCTION(LATEX_COPY_INPUT_FILE)

#############################################################################
# Commands provided by the UseLATEX.cmake "package"
#############################################################################

FUNCTION(LATEX_USAGE command message)
  MESSAGE(SEND_ERROR
    "${message}\nUsage: ${command}(<tex_file>\n           [BIBFILES <bib_file> <bib_file> ...]\n           [INPUTS <tex_file> <tex_file> ...]\n           [IMAGE_DIRS <directory1> <directory2> ...]\n           [IMAGES <image_file1> <image_file2>\n           [CONFIGURE <tex_file> <tex_file> ...]\n           [DEPENDS <tex_file> <tex_file> ...]\n           [MULTIBIB_NEWCITES] <suffix_list>\n           [USE_INDEX] [USE_GLOSSARY] [USE_NOMENCL]\n           [DEFAULT_PDF] [DEFAULT_SAFEPDF]\n           [MANGLE_TARGET_NAMES])"
    )
ENDFUNCTION(LATEX_USAGE command message)

# Parses arguments to ADD_LATEX_DOCUMENT and ADD_LATEX_TARGETS and sets the
# variables LATEX_TARGET, LATEX_IMAGE_DIR, LATEX_BIBFILES, LATEX_DEPENDS, and
# LATEX_INPUTS.
FUNCTION(PARSE_ADD_LATEX_ARGUMENTS command)
  LATEX_PARSE_ARGUMENTS(
    LATEX
    "BIBFILES;MULTIBIB_NEWCITES;INPUTS;IMAGE_DIRS;IMAGES;CONFIGURE;DEPENDS"
    "USE_INDEX;USE_GLOSSARY;USE_GLOSSARIES;USE_NOMENCL;DEFAULT_PDF;DEFAULT_SAFEPDF;MANGLE_TARGET_NAMES"
    ${ARGN}
    )

  # The first argument is the target latex file.
  IF (LATEX_DEFAULT_ARGS)
    LIST(GET LATEX_DEFAULT_ARGS 0 latex_main_input)
    LIST(REMOVE_AT LATEX_DEFAULT_ARGS 0)
    LATEX_GET_FILENAME_COMPONENT(latex_target ${latex_main_input} NAME_WE)
    SET(LATEX_MAIN_INPUT ${latex_main_input} CACHE INTERNAL "" FORCE)
    SET(LATEX_TARGET ${latex_target} CACHE INTERNAL "" FORCE)
  ELSE (LATEX_DEFAULT_ARGS)
    LATEX_USAGE(${command} "No tex file target given to ${command}.")
  ENDIF (LATEX_DEFAULT_ARGS)

  IF (LATEX_DEFAULT_ARGS)
    LATEX_USAGE(${command} "Invalid or depricated arguments: ${LATEX_DEFAULT_ARGS}")
  ENDIF (LATEX_DEFAULT_ARGS)

  # Backward compatibility between 1.6.0 and 1.6.1.
  IF (LATEX_USE_GLOSSARIES)
    SET(LATEX_USE_GLOSSARY TRUE CACHE INTERNAL "" FORCE)
  ENDIF (LATEX_USE_GLOSSARIES)
ENDFUNCTION(PARSE_ADD_LATEX_ARGUMENTS)

FUNCTION(ADD_LATEX_TARGETS_INTERNAL)
  IF (LATEX_USE_SYNCTEX)
    SET(synctex_flags ${LATEX_SYNCTEX_FLAGS})
  ELSE (LATEX_USE_SYNCTEX)
    SET(synctex_flags)
  ENDIF (LATEX_USE_SYNCTEX)

  # The commands to run LaTeX.  They are repeated multiple times.
  SET(latex_build_command
    ${LATEX_COMPILER} ${LATEX_COMPILER_FLAGS} ${synctex_flags} ${LATEX_MAIN_INPUT}
    )
  SET(pdflatex_build_command
    ${PDFLATEX_COMPILER} ${PDFLATEX_COMPILER_FLAGS} ${synctex_flags} ${LATEX_MAIN_INPUT}
    )

  # Set up target names.
  IF (LATEX_MANGLE_TARGET_NAMES)
    SET(dvi_target      ${LATEX_TARGET}_dvi)
    SET(pdf_target      ${LATEX_TARGET}_pdf)
    SET(ps_target       ${LATEX_TARGET}_ps)
    SET(safepdf_target  ${LATEX_TARGET}_safepdf)
    SET(html_target     ${LATEX_TARGET}_html)
    SET(auxclean_target ${LATEX_TARGET}_auxclean)
  ELSE (LATEX_MANGLE_TARGET_NAMES)
    SET(dvi_target      dvi)
    SET(pdf_target      pdf)
    SET(ps_target       ps)
    SET(safepdf_target  safepdf)
    SET(html_target     html)
    SET(auxclean_target auxclean)
  ENDIF (LATEX_MANGLE_TARGET_NAMES)

  # Probably not all of these will be generated, but they could be.
  # Note that the aux file is added later.
  SET(auxiliary_clean_files
    ${output_dir}/${LATEX_TARGET}.bbl
    ${output_dir}/${LATEX_TARGET}.blg
    ${output_dir}/${LATEX_TARGET}-blx.bib
    ${output_dir}/${LATEX_TARGET}.glg
    ${output_dir}/${LATEX_TARGET}.glo
    ${output_dir}/${LATEX_TARGET}.gls
    ${output_dir}/${LATEX_TARGET}.idx
    ${output_dir}/${LATEX_TARGET}.ilg
    ${output_dir}/${LATEX_TARGET}.ind
    ${output_dir}/${LATEX_TARGET}.ist
    ${output_dir}/${LATEX_TARGET}.log
    ${output_dir}/${LATEX_TARGET}.out
    ${output_dir}/${LATEX_TARGET}.toc
    ${output_dir}/${LATEX_TARGET}.lof
    ${output_dir}/${LATEX_TARGET}.xdy
    ${output_dir}/${LATEX_TARGET}.synctex.gz
    ${output_dir}/${LATEX_TARGET}.synctex.bak.gz
    ${output_dir}/${LATEX_TARGET}.dvi
    ${output_dir}/${LATEX_TARGET}.ps
    ${output_dir}/${LATEX_TARGET}.pdf
    )

  SET(image_list ${LATEX_IMAGES})

  # For each directory in LATEX_IMAGE_DIRS, glob all the image files and
  # place them in LATEX_IMAGES.
  FOREACH(dir ${LATEX_IMAGE_DIRS})
    IF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
      MESSAGE(WARNING "Image directory ${CMAKE_CURRENT_SOURCE_DIR}/${dir} does not exist.  Are you sure you gave relative directories to IMAGE_DIRS?")
    ENDIF (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
    FOREACH(extension ${LATEX_IMAGE_EXTENSIONS})
      FILE(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*${extension})
      FOREACH(file ${files})
        LATEX_GET_FILENAME_COMPONENT(filename ${file} NAME)
        SET(image_list ${image_list} ${dir}/${filename})
      ENDFOREACH(file)
    ENDFOREACH(extension)
  ENDFOREACH(dir)

  LATEX_PROCESS_IMAGES(dvi_images pdf_images ${image_list})

  SET(make_dvi_command
    ${CMAKE_COMMAND} -E chdir ${output_dir}
    ${latex_build_command})
  SET(make_pdf_command
    ${CMAKE_COMMAND} -E chdir ${output_dir}
    ${pdflatex_build_command}
    )

  SET(make_dvi_depends ${LATEX_DEPENDS} ${dvi_images})
  SET(make_pdf_depends ${LATEX_DEPENDS} ${pdf_images})
  FOREACH(input ${LATEX_MAIN_INPUT} ${LATEX_INPUTS})
    SET(make_dvi_depends ${make_dvi_depends} ${output_dir}/${input})
    SET(make_pdf_depends ${make_pdf_depends} ${output_dir}/${input})
    IF (${input} MATCHES "\\.tex$")
      STRING(REGEX REPLACE "\\.tex$" "" input_we ${input})
      SET(auxiliary_clean_files ${auxiliary_clean_files}
        ${output_dir}/${input_we}.aux
        ${output_dir}/${input}.aux
        )
    ENDIF (${input} MATCHES "\\.tex$")
  ENDFOREACH(input)

  IF (LATEX_USE_GLOSSARY)
    FOREACH(dummy 0 1)   # Repeat these commands twice.
      SET(make_dvi_command ${make_dvi_command}
        COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${CMAKE_COMMAND}
        -D LATEX_BUILD_COMMAND=makeglossaries
        -D LATEX_TARGET=${LATEX_TARGET}
        -D MAKEINDEX_COMPILER=${MAKEINDEX_COMPILER}
        -D XINDY_COMPILER=${XINDY_COMPILER}
        -D MAKEGLOSSARIES_COMPILER_FLAGS=${MAKEGLOSSARIES_COMPILER_FLAGS}
        -P ${LATEX_USE_LATEX_LOCATION}
        COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${latex_build_command}
        )
      SET(make_pdf_command ${make_pdf_command}
        COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${CMAKE_COMMAND}
        -D LATEX_BUILD_COMMAND=makeglossaries
        -D LATEX_TARGET=${LATEX_TARGET}
        -D MAKEINDEX_COMPILER=${MAKEINDEX_COMPILER}
        -D XINDY_COMPILER=${XINDY_COMPILER}
        -D MAKEGLOSSARIES_COMPILER_FLAGS=${MAKEGLOSSARIES_COMPILER_FLAGS}
        -P ${LATEX_USE_LATEX_LOCATION}
        COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${pdflatex_build_command}
        )
    ENDFOREACH(dummy)
  ENDIF (LATEX_USE_GLOSSARY)

  IF (LATEX_USE_NOMENCL)
    FOREACH(dummy 0 1)   # Repeat these commands twice.
      SET(make_dvi_command ${make_dvi_command}
        COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${CMAKE_COMMAND}
        -D LATEX_BUILD_COMMAND=makenomenclature
        -D LATEX_TARGET=${LATEX_TARGET}
        -D MAKEINDEX_COMPILER=${MAKEINDEX_COMPILER}
        -D MAKENOMENCLATURE_COMPILER_FLAGS=${MAKENOMENCLATURE_COMPILER_FLAGS}
        -P ${LATEX_USE_LATEX_LOCATION}
        COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${latex_build_command}
        )
      SET(make_pdf_command ${make_pdf_command}
        COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${CMAKE_COMMAND}
        -D LATEX_BUILD_COMMAND=makenomenclature
        -D LATEX_TARGET=${LATEX_TARGET}
        -D MAKEINDEX_COMPILER=${MAKEINDEX_COMPILER}
        -D MAKENOMENCLATURE_COMPILER_FLAGS=${MAKENOMENCLATURE_COMPILER_FLAGS}
        -P ${LATEX_USE_LATEX_LOCATION}
        COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${pdflatex_build_command}
        )
    ENDFOREACH(dummy)
  ENDIF (LATEX_USE_NOMENCL)

  IF (LATEX_BIBFILES)
    IF (LATEX_MULTIBIB_NEWCITES)
      FOREACH (multibib_auxfile ${LATEX_MULTIBIB_NEWCITES})
        LATEX_GET_FILENAME_COMPONENT(multibib_target ${multibib_auxfile} NAME_WE)
        SET(make_dvi_command ${make_dvi_command}
          COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
          ${BIBTEX_COMPILER} ${BIBTEX_COMPILER_FLAGS} ${multibib_target})
        SET(make_pdf_command ${make_pdf_command}
          COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
          ${BIBTEX_COMPILER} ${BIBTEX_COMPILER_FLAGS} ${multibib_target})
        SET(auxiliary_clean_files ${auxiliary_clean_files}
          ${output_dir}/${multibib_target}.aux)
      ENDFOREACH (multibib_auxfile ${LATEX_MULTIBIB_NEWCITES})
    ELSE (LATEX_MULTIBIB_NEWCITES)
      SET(make_dvi_command ${make_dvi_command}
        COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${BIBTEX_COMPILER} ${BIBTEX_COMPILER_FLAGS} ${LATEX_TARGET})
      SET(make_pdf_command ${make_pdf_command}
        COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${BIBTEX_COMPILER} ${BIBTEX_COMPILER_FLAGS} ${LATEX_TARGET})
    ENDIF (LATEX_MULTIBIB_NEWCITES)

    FOREACH (bibfile ${LATEX_BIBFILES})
      SET(make_dvi_depends ${make_dvi_depends} ${output_dir}/${bibfile})
      SET(make_pdf_depends ${make_pdf_depends} ${output_dir}/${bibfile})
    ENDFOREACH (bibfile ${LATEX_BIBFILES})
  ELSE (LATEX_BIBFILES)
    IF (LATEX_MULTIBIB_NEWCITES)
      MESSAGE(WARNING "MULTIBIB_NEWCITES has no effect without BIBFILES option.")
    ENDIF (LATEX_MULTIBIB_NEWCITES)
  ENDIF (LATEX_BIBFILES)

  IF (LATEX_USE_INDEX)
    SET(make_dvi_command ${make_dvi_command}
      COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
      ${latex_build_command}
      COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
      ${MAKEINDEX_COMPILER} ${MAKEINDEX_COMPILER_FLAGS} ${LATEX_TARGET}.idx)
    SET(make_pdf_command ${make_pdf_command}
      COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
      ${pdflatex_build_command}
      COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
      ${MAKEINDEX_COMPILER} ${MAKEINDEX_COMPILER_FLAGS} ${LATEX_TARGET}.idx)
  ENDIF (LATEX_USE_INDEX)

  SET(make_dvi_command ${make_dvi_command}
    COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
    ${latex_build_command}
    COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
    ${latex_build_command})
  SET(make_pdf_command ${make_pdf_command}
    COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
    ${pdflatex_build_command}
    COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
    ${pdflatex_build_command})

  IF (LATEX_USE_SYNCTEX)
    IF (NOT GZIP)
      MESSAGE(SEND_ERROR "UseLATEX.cmake: USE_SYNTEX option requires gzip program.  Set GZIP variable.")
    ENDIF (NOT GZIP)
    SET(make_dvi_command ${make_dvi_command}
      COMMAND ${CMAKE_COMMAND}
      -D LATEX_BUILD_COMMAND=correct_synctex
      -D LATEX_TARGET=${LATEX_TARGET}
      -D GZIP=${GZIP}
      -D "LATEX_SOURCE_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}"
      -D "LATEX_BINARY_DIRECTORY=${output_dir}"
      -P ${LATEX_USE_LATEX_LOCATION}
      )
    SET(make_pdf_command ${make_pdf_command}
      COMMAND ${CMAKE_COMMAND}
      -D LATEX_BUILD_COMMAND=correct_synctex
      -D LATEX_TARGET=${LATEX_TARGET}
      -D GZIP=${GZIP}
      -D "LATEX_SOURCE_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}"
      -D "LATEX_BINARY_DIRECTORY=${output_dir}"
      -P ${LATEX_USE_LATEX_LOCATION}
      )
  ENDIF (LATEX_USE_SYNCTEX)

  # Add commands and targets for building dvi outputs.
  ADD_CUSTOM_COMMAND(OUTPUT ${output_dir}/${LATEX_TARGET}.dvi
    COMMAND ${make_dvi_command}
    DEPENDS ${make_dvi_depends}
    )
  IF (LATEX_DEFAULT_PDF OR LATEX_DEFAULT_SAFEPDF)
    ADD_CUSTOM_TARGET(${dvi_target}
      DEPENDS ${output_dir}/${LATEX_TARGET}.dvi)
  ELSE (LATEX_DEFAULT_PDF OR LATEX_DEFAULT_SAFEPDF)
    ADD_CUSTOM_TARGET(${dvi_target} ALL
      DEPENDS ${output_dir}/${LATEX_TARGET}.dvi)
  ENDIF (LATEX_DEFAULT_PDF OR LATEX_DEFAULT_SAFEPDF)

  # Add commands and targets for building pdf outputs (with pdflatex).
  IF (PDFLATEX_COMPILER)
    ADD_CUSTOM_COMMAND(OUTPUT ${output_dir}/${LATEX_TARGET}.pdf
      COMMAND ${make_pdf_command}
      DEPENDS ${make_pdf_depends}
      )
    IF (LATEX_DEFAULT_PDF)
      ADD_CUSTOM_TARGET(${pdf_target} ALL
        DEPENDS ${output_dir}/${LATEX_TARGET}.pdf)
    ELSE (LATEX_DEFAULT_PDF)
      ADD_CUSTOM_TARGET(${pdf_target}
        DEPENDS ${output_dir}/${LATEX_TARGET}.pdf)
    ENDIF (LATEX_DEFAULT_PDF)
  ENDIF (PDFLATEX_COMPILER)

  IF (DVIPS_CONVERTER)
    ADD_CUSTOM_COMMAND(OUTPUT ${output_dir}/${LATEX_TARGET}.ps
      COMMAND ${CMAKE_COMMAND} -E chdir ${output_dir}
        ${DVIPS_CONVERTER} ${DVIPS_CONVERTER_FLAGS} -o ${LATEX_TARGET}.ps ${LATEX_TARGET}.dvi
      DEPENDS ${output_dir}/${LATEX_TARGET}.dvi)
    ADD_CUSTOM_TARGET(${ps_target}
      DEPENDS ${output_dir}/${LATEX_TARGET}.ps)
    IF (PS2PDF_CONVERTER)
      # Since both the pdf and safepdf targets have the same output, we
      # cannot properly do the dependencies for both.  When selecting safepdf,
      # simply force a recompile every time.
      IF (LATEX_DEFAULT_SAFEPDF)
        ADD_CUSTOM_TARGET(${safepdf_target} ALL
          ${CMAKE_COMMAND} -E chdir ${output_dir}
          ${PS2PDF_CONVERTER} ${PS2PDF_CONVERTER_FLAGS} ${LATEX_TARGET}.ps ${LATEX_TARGET}.pdf
          )
      ELSE (LATEX_DEFAULT_SAFEPDF)
        ADD_CUSTOM_TARGET(${safepdf_target}
          ${CMAKE_COMMAND} -E chdir ${output_dir}
          ${PS2PDF_CONVERTER} ${PS2PDF_CONVERTER_FLAGS} ${LATEX_TARGET}.ps ${LATEX_TARGET}.pdf
          )
      ENDIF (LATEX_DEFAULT_SAFEPDF)
      ADD_DEPENDENCIES(${safepdf_target} ${ps_target})
    ENDIF (PS2PDF_CONVERTER)
  ENDIF (DVIPS_CONVERTER)

  IF (LATEX2HTML_CONVERTER)
    ADD_CUSTOM_TARGET(${html_target}
      ${CMAKE_COMMAND} -E chdir ${output_dir}
      ${LATEX2HTML_CONVERTER} ${LATEX2HTML_CONVERTER_FLAGS} ${LATEX_MAIN_INPUT}
      )
    ADD_DEPENDENCIES(${html_target} ${LATEX_MAIN_INPUT} ${LATEX_INPUTS})
  ENDIF (LATEX2HTML_CONVERTER)

  SET_DIRECTORY_PROPERTIES(.
    ADDITIONAL_MAKE_CLEAN_FILES "${auxiliary_clean_files}"
    )

  ADD_CUSTOM_TARGET(${auxclean_target}
    COMMENT "Cleaning auxiliary LaTeX files."
    COMMAND ${CMAKE_COMMAND} -E remove ${auxiliary_clean_files}
    )
ENDFUNCTION(ADD_LATEX_TARGETS_INTERNAL)

FUNCTION(ADD_LATEX_TARGETS)
  LATEX_GET_OUTPUT_PATH(output_dir)
  PARSE_ADD_LATEX_ARGUMENTS(ADD_LATEX_TARGETS ${ARGV})

  ADD_LATEX_TARGETS_INTERNAL()
ENDFUNCTION(ADD_LATEX_TARGETS)

FUNCTION(ADD_LATEX_DOCUMENT)
  LATEX_GET_OUTPUT_PATH(output_dir)
  IF (output_dir)
    PARSE_ADD_LATEX_ARGUMENTS(ADD_LATEX_DOCUMENT ${ARGV})

    LATEX_COPY_INPUT_FILE(${LATEX_MAIN_INPUT})

    FOREACH (bib_file ${LATEX_BIBFILES})
      LATEX_COPY_INPUT_FILE(${bib_file})
    ENDFOREACH (bib_file)

    FOREACH (input ${LATEX_INPUTS})
      LATEX_COPY_INPUT_FILE(${input})
    ENDFOREACH(input)

    LATEX_COPY_GLOBBED_FILES(${CMAKE_CURRENT_SOURCE_DIR}/*.cls ${output_dir})
    LATEX_COPY_GLOBBED_FILES(${CMAKE_CURRENT_SOURCE_DIR}/*.bst ${output_dir})
    LATEX_COPY_GLOBBED_FILES(${CMAKE_CURRENT_SOURCE_DIR}/*.clo ${output_dir})
    LATEX_COPY_GLOBBED_FILES(${CMAKE_CURRENT_SOURCE_DIR}/*.sty ${output_dir})
    LATEX_COPY_GLOBBED_FILES(${CMAKE_CURRENT_SOURCE_DIR}/*.ist ${output_dir})

    ADD_LATEX_TARGETS_INTERNAL()
  ENDIF (output_dir)
ENDFUNCTION(ADD_LATEX_DOCUMENT)

#############################################################################
# Actually do stuff
#############################################################################

IF (LATEX_BUILD_COMMAND)
  SET(command_handled)

  IF ("${LATEX_BUILD_COMMAND}" STREQUAL makeglossaries)
    LATEX_MAKEGLOSSARIES()
    SET(command_handled TRUE)
  ENDIF ("${LATEX_BUILD_COMMAND}" STREQUAL makeglossaries)

  IF ("${LATEX_BUILD_COMMAND}" STREQUAL makenomenclature)
    LATEX_MAKENOMENCLATURE()
    SET(command_handled TRUE)
  ENDIF ("${LATEX_BUILD_COMMAND}" STREQUAL makenomenclature)

  IF ("${LATEX_BUILD_COMMAND}" STREQUAL correct_synctex)
    LATEX_CORRECT_SYNCTEX()
    SET(command_handled TRUE)
  ENDIF ("${LATEX_BUILD_COMMAND}" STREQUAL correct_synctex)

  IF (NOT command_handled)
    MESSAGE(SEND_ERROR "Unknown command: ${LATEX_BUILD_COMMAND}")
  ENDIF (NOT command_handled)

ELSE (LATEX_BUILD_COMMAND)
  # Must be part of the actual configure (included from CMakeLists.txt).
  LATEX_SETUP_VARIABLES()
ENDIF (LATEX_BUILD_COMMAND)
