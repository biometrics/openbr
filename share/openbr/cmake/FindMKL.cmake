find_path(MKL_DIR include/mkl_cblas.h /opt/intel/mkl)
include_directories(${MKL_DIR}/include)
set(MKL_LIBS ${MKL_DIR}/lib/libmkl_rt.dylib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_DIR)
