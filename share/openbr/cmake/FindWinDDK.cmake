find_path(WINDDK_DIR project.mk C:/WinDDK/*)
include_directories(${WINDDK_DIR}/inc/api)
add_custom_target(REMOVE_SAL ALL cmake -E remove -f ${WINDDK_DIR}/inc/api/sal.h)

if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
  set(WINDDK_LIBS ${WINDDK_DIR}/lib/win7/amd64/hid.lib ${WINDDK_DIR}/lib/win7/amd64/setupapi.lib ${WINDDK_DIR}/lib/win7/amd64/uuid.lib)
else()
  set(WINDDK_LIBS ${WINDDK_DIR}/lib/win7/i386/hid.lib ${WINDDK_DIR}/lib/win7/i386/setupapi.lib)
endif()
