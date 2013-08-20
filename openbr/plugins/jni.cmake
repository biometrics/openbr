set(BR_WITH_JAVA ON CACHE BOOL "Use Java Code")

if (${BR_WITH_JAVA})
	find_package(JNI REQUIRED)
        find_package(JAVA REQUIRED)
	set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/jni.cpp)
	set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${JNI_LIBRARIES})

	include_directories(${JAVA_INCLUDE_PATH})
	include_directories(${JAVA_INCLUDE_PATH2})

        #include(UseJava)
        #file(GLOB JAVA_CLASS_FILES ${BR_SHARE_DIR}/Java/*.class)
        #add_jar(jni ${JAVA_CLASS_FILES})
        #install(FILES ${jni} DESTINATION .)
endif()
