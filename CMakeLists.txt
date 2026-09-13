cmake_minimum_required(VERSION 3.24)

# -- Supported platform -------------------------------------------------------------------------
# Wheels exist for Linux (x86-64 and aarch64), Windows x64 and macOS arm64; anywhere else pip falls
# back to THIS source distribution, and the build dies inside project() with "No CMAKE_CXX_COMPILER
# could be found" -- a message about a missing compiler when the real answer is usually "not this
# operating system". Say the real answer first. A source build on a supported platform is fine and
# is exactly what the sdist is for; it is the unknown ones that get stopped here.
# (History: a Windows user met the CMake message on 2026-09-13; the wheel probe that followed showed
#  Windows and macOS build cleanly, so they became supported rather than rejected. Linux keeps the
#  OpenMP host backend; Windows and macOS get Serial, because AppleClang ships no OpenMP and MSVC
#  reports 2.0 whatever runtime is selected. suite/docs/RELEASE_PREP.md section 11.1.)
cmake_host_system_information(RESULT PECLET_HOST_OS QUERY OS_NAME)
if(NOT PECLET_HOST_OS MATCHES "^(Linux|Windows|macOS|Darwin)$" AND NOT PECLET_ALLOW_UNSUPPORTED_PLATFORM)
  message(FATAL_ERROR
    "peclet-coupling is built and tested on Linux, Windows and macOS; this host reports ${PECLET_HOST_OS}.\n"
    "  Nothing here is known to be wrong with your platform -- it has simply never been tried.\n"
    "  To try it anyway, configure with -DPECLET_ALLOW_UNSUPPORTED_PLATFORM=ON and tell us how it\n"
    "  went: https://github.com/computational-chemical-engineering/peclet/issues\n"
    "  To run peclet right now with no install at all, open the quick start in a browser --\n"
    "    https://colab.research.google.com/github/computational-chemical-engineering/peclet/blob/main/docs/notebooks/quickstart_sphere.ipynb")
endif()

project(peclet_coupling LANGUAGES CXX)

# peclet.coupling -- CFD-DEM exchange kernels (particle<->grid deposition, drag, momentum feedback)
# as an importable Python module. Physics-free glue: the Python CfdDem driver (python/peclet_coupling)
# composes peclet.flow + peclet.dem, and this module runs the kernels in place on the arrays they
# expose. Header-only compute (core interp primitive + drag laws), Kokkos backend from the install
# prefix (extern/install/<backend>), nanobind via the shared SuiteNanobind helper.
#
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp"
#   cmake --build build -j        ->  build/peclet/coupling/_coupling.*.so

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Dependencies via the vendored PecletDeps helper: an installed Kokkos prefix + sibling checkout for
# the dev/suite build, or FetchContent-built Kokkos + fetched core headers (v-pinned) for a
# self-contained sdist build (`pip install peclet-coupling`). See cmake/PecletDeps.cmake.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PecletDeps)
peclet_require_kokkos()
peclet_require_nanobind()
peclet_sibling_include(peclet-core "${PECLET_CORE_TAG}" "../core" CORE_INC)

nanobind_add_module(coupling NB_STATIC NOMINSIZE src/coupling_bindings.cpp)
set_target_properties(coupling PROPERTIES OUTPUT_NAME _coupling  # -> peclet.coupling._coupling
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/coupling")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/peclet_coupling/__init__.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/coupling/__init__.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/peclet_coupling/driver.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/coupling/driver.py COPYONLY)
# resolved.py is imported by __init__.py, so omitting it makes a dev build unimportable.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/peclet_coupling/resolved.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/coupling/resolved.py COPYONLY)
target_include_directories(coupling PRIVATE src "${CORE_INC}")
target_link_libraries(coupling PRIVATE Kokkos::kokkos)
if(Kokkos_ENABLE_HIP)
  target_link_options(coupling PRIVATE -Wl,--no-gc-sections)
  # HIP/lld experiment 1 (docs/RELEASE.md §8): nanobind sets CXX_VISIBILITY_PRESET hidden, and under hipcc
  # the host objects then reference the Kokkos SharedAllocationRecord / shared_ptr control-block vtables
  # as hidden symbols that nothing defines ("undefined hidden symbol: vtable for ..."). Default visibility
  # on the HIP path only; CUDA/OpenMP builds are untouched.
  set_target_properties(coupling PROPERTIES CXX_VISIBILITY_PRESET default)
endif()

option(PECLET_COUPLING_TESTS "Register the Python CFD-DEM integration tests as ctests" OFF)
if(PECLET_COUPLING_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# --- pip / scikit-build-core install rule --------------------------------------------------------
# Places the extension + the peclet/coupling package (__init__.py + driver.py + resolved.py) into the PEP-420
# peclet namespace. In an sdist build the core headers + nanobind come from PecletDeps (fetched at
# the pinned tag), so the source distribution is self-contained.
if(DEFINED SKBUILD)
  install(TARGETS coupling LIBRARY DESTINATION peclet/coupling COMPONENT python)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/peclet_coupling/__init__.py
                ${CMAKE_CURRENT_SOURCE_DIR}/python/peclet_coupling/driver.py
                ${CMAKE_CURRENT_SOURCE_DIR}/python/peclet_coupling/resolved.py
          DESTINATION peclet/coupling COMPONENT python)
endif()
