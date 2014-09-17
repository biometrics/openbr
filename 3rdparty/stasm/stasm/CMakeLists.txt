# Stasm CMakeLists

aux_source_directory(stasm/src STASM_SRC)
include_directories(stasm/include)

#aux_source_directory(tasm/src TASM_SRC)
#include_directories(tasm/include)

add_library(stasm SHARED ${STASM_SRC} ${TASM_SRC})
qt5_use_modules(stasm ${QT_DEPENDENCIES})

set_target_properties(stasm PROPERTIES
                                   DEFINE_SYMBOL STASM_LIBRARY
                                   VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}
                                   SOVERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}
                                   LINK_INTERFACE_LIBRARIES ""
                                   AUTOMOC TRUE)

target_link_libraries(stasm ${OpenCV_LIBS} ${Qt5Core_QTMAIN_LIBRARIES})

install(DIRECTORY stasm/include/ DESTINATION include/stasm)
install(DIRECTORY tasm/include/ DESTINATION include/tasm)

install(TARGETS stasm RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
