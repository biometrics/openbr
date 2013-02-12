set(BR_WITH_LLVM OFF CACHE BOOL "Build LLVM features")

if(${BR_WITH_LLVM})
  # A convenience variable:
  set(LLVM_ROOT "/usr/local" CACHE PATH "Root of LLVM install.")

  # A bit of a sanity check:
  if (NOT EXISTS ${LLVM_ROOT}/include/llvm)
    message(FATAL_ERROR "LLVM_ROOT (${LLVM_ROOT}) is not a valid LLVM install")
  endif()

  # We incorporate the CMake features provided by LLVM:
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${LLVM_ROOT}/share/llvm/cmake")
  include(LLVMConfig)

  # Now set the header and library paths:
  include_directories(${LLVM_INCLUDE_DIRS})
  link_directories(${LLVM_LIBRARY_DIRS})
  add_definitions(${LLVM_DEFINITIONS})

  # Let's suppose we want to build a JIT compiler with support for binary code:
  llvm_map_components_to_libraries(REQ_LLVM_LIBRARIES jit native)

  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC}
                        ${CMAKE_SOURCE_DIR}/sdk/plugins/llvm.cpp
                        ${CMAKE_SOURCE_DIR}/sdk/likely/likely_internal.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${REQ_LLVM_LIBRARIES})
endif()
