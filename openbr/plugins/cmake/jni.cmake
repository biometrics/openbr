set(BR_WITH_JAVA OFF CACHE BOOL "Use Java Code")

if (${BR_WITH_JAVA})
	find_package(JNI REQUIRED)
        find_package(JAVA REQUIRED)
	set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${JNI_LIBRARIES})

	include_directories(${JAVA_INCLUDE_PATH})
	include_directories(${JAVA_INCLUDE_PATH2})
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/core/jni.cpp)
endif()
