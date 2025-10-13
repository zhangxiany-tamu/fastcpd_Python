cmake_minimum_required(VERSION 3.18...3.27)

project(fastcpd LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Execute nanobind's CMake helper
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
  RESULT_VARIABLE NB_RESULT
)
if(NOT NB_RESULT EQUAL 0)
  message(FATAL_ERROR "nanobind package not found. Install with: pip install nanobind")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Find Armadillo (linear algebra library)
find_package(Armadillo REQUIRED)
if(NOT ARMADILLO_FOUND)
  message(FATAL_ERROR "Armadillo not found. Install with: brew install armadillo (macOS) or apt-get install libarmadillo-dev (Linux)")
endif()

include_directories(${ARMADILLO_INCLUDE_DIRS})

# Find BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Define compile-time options
# - NO_RCPP: compile without any R dependencies
# - ARMA_DONT_USE_WRAPPER: use Armadillo in header-only mode (no libarmadillo linkage)
# - ARMA_USE_BLAS/LAPACK: enable BLAS/LAPACK calls directly
add_compile_definitions(NO_RCPP ARMA_DONT_USE_WRAPPER ARMA_USE_BLAS ARMA_USE_LAPACK)

# Compiler flags for optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Avoid -march=native to produce portable wheels (manylinux compatible)
  add_compile_options(-O3 -Wall -Wextra)
  # OpenMP is optional - disable on Apple Clang due to compatibility issues
  if(NOT APPLE OR NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
      add_compile_options(${OpenMP_CXX_FLAGS})
      link_libraries(${OpenMP_CXX_LIBRARIES})
    endif()
  endif()
elseif(MSVC)
  add_compile_options(/O2 /W4)
endif()

# Source files
# Note: GLM/LASSO models use sklearn (faster than R), so we exclude those C++ files
# Note: GARCH/ARMA models temporarily excluded due to R dependencies
# Note: fastcpd_impl.cc is included directly in bindings.cc, so not listed here
set(FASTCPD_SOURCES
    src/fastcpd.cc
    # src/fastcpd_impl.cc               # Included in bindings.cc
    # src/ref_fastglm_fit_glm.cc        # Excluded - uses Rcpp
    # src/ref_fastglm_fit_glm_dense.cc  # Excluded - uses Rcpp
    # src/ref_tseries.cc                # Excluded - uses Rcpp
    src/ref_family_python.cc
    src/ref_tseries_formats_stub.cc
    src/bindings.cc
)

# Fortran sources for GARCH optimization
# Temporarily excluded - need to fix tseries dependencies first
set(FORTRAN_SOURCES
    # src/ref_tseries_dsumsl.f
    # src/ref_tseries_cfuncs.f90
)

# Create Python extension module
nanobind_add_module(
  _fastcpd_impl
  STABLE_ABI
  NB_STATIC
  ${FASTCPD_SOURCES}
  ${FORTRAN_SOURCES}
)

# Link libraries (link directly against BLAS/LAPACK; avoid libarmadillo to keep portability)
target_link_libraries(_fastcpd_impl PRIVATE
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
)

target_include_directories(_fastcpd_impl PRIVATE src)

# Fix rpath for runtime library loading (especially OpenBLAS)
if(APPLE)
  set_target_properties(_fastcpd_impl PROPERTIES
    INSTALL_RPATH "@loader_path;/opt/homebrew/lib;/usr/local/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()

if(UNIX AND NOT APPLE)
  # Ensure the wheel's bundled libs can be found at runtime on Linux
  # auditwheel places vendored libs under .libs next to the extension
  set_target_properties(_fastcpd_impl PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/.libs"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()

# Help auditwheel/auditing tools discover and fix up linked libs on Linux
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Install the extension module using scikit-build-core convention
# For Python extension modules, we need to handle different platforms:
# - Linux/macOS: .so files use LIBRARY
# - Windows: .pyd files use RUNTIME
install(
  TARGETS _fastcpd_impl
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/fastcpd
  RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/fastcpd
)
