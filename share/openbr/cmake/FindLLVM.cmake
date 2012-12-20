find_path(LLVM_DIR LLVMBuild.txt ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(LLVM_DIR)
include_directories(${LLVM_DIR}/include)
set(LLVM_LICENSE ${LLVM_DIR}/LICENSE.TXT)

