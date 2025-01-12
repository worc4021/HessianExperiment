set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
cmake_minimum_required(VERSION 3.20)
project(scan LANGUAGES CXX)

find_package(Eigen3 CONFIG REQUIRED)
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)
set(MKL_LINK static)
set(MKL_THREADING sequential)
set(MKL_INTERFACE lp64)
find_package(MKL REQUIRED)
include(FindOpenMP)

include(FetchContent)
set(AUTODIFF_BUILD_TESTS OFF)
set(AUTODIFF_BUILD_PYTHON OFF)
set(AUTODIFF_BUILD_EXAMPLES OFF)
set(AUTODIFF_BUILD_DOCS OFF)
FetchContent_Declare(
  autodiff
  GIT_REPOSITORY https://github.com/autodiff/autodiff.git
  GIT_TAG        v1.1.2
)
FetchContent_MakeAvailable(autodiff)

set(GLOBAL_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Ipopt
option(IPOPT_PREFIX "Path to the IPOPT installation" "")
if (NOT IPOPT_PREFIX)
    message(FATAL_ERROR "IPOPT_PREFIX must be set to the path of the IPOPT installation")
else()
    if (WIN32)
        set(ENV{PKG_CONFIG_PATH} "${IPOPT_PREFIX}/lib/pkgconfig;$ENV{PKG_CONFIG_PATH}")
    else()
        set(ENV{PKG_CONFIG_PATH} "${IPOPT_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    endif()
endif()
add_subdirectory(ipopt)

# Uno
include($ENV{CMAKE_INSTALL_PREFIX}/cmake/GKlibTargets.cmake)
include($ENV{CMAKE_INSTALL_PREFIX}/cmake/metisTargets.cmake)
include($ENV{CMAKE_INSTALL_PREFIX}/cmake/bqpdTargets.cmake)
include($ENV{CMAKE_INSTALL_PREFIX}/cmake/mumpsTargets.cmake)
include($ENV{CMAKE_INSTALL_PREFIX}/cmake/coinhslTargets.cmake)
include($ENV{CMAKE_INSTALL_PREFIX}/cmake/unoTargets.cmake)
add_subdirectory(uno)
