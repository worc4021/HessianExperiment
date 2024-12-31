set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
cmake_minimum_required(VERSION 3.20)
project(scan LANGUAGES CXX)

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

find_package(PkgConfig REQUIRED)
pkg_check_modules(ipopt REQUIRED IMPORTED_TARGET ipopt)


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

set(MKL_LINK "static")
set(MKL_THREADING "sequential")
set(MKL_INTERFACE_FULL "intel_lp64")
find_package(MKL REQUIRED)

find_package(Eigen3 CONFIG REQUIRED)
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)
include(FindOpenMP)

set(COINIPOPTLIBS)
if (WIN32)
    set(CMAKE_CXX_FLAGS "/Wall /EHsc")
    list(APPEND COINIPOPTLIBS ${ipopt_LIBRARY_DIRS}/ipopt.lib ${ipopt_LIBRARY_DIRS}/coinhsl.lib ${ipopt_LIBRARY_DIRS}/coinmumps.lib ${ipopt_LIBRARY_DIRS}/libcoinmetis.lib)
else()
    set(CMAKE_CXX_FLAGS "-fpie")
    list(APPEND COINIPOPTLIBS ${ipopt_LIBRARY_DIRS}/libipopt.a ${ipopt_LIBRARY_DIRS}/libcoinhsl.a ${ipopt_LIBRARY_DIRS}/libcoinmumps.a ${ipopt_LIBRARY_DIRS}/libcoinmetis.a)
endif()


add_executable(ipopt_scan src/main.cpp src/nlp.cpp)

target_include_directories(ipopt_scan PRIVATE 
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${ipopt_INCLUDE_DIRS})

# target_link_directories(ipopt_scan PRIVATE ${ipopt_LIBRARY_DIRS})
target_link_libraries(ipopt_scan PRIVATE PkgConfig::ipopt autodiff::autodiff Arrow::arrow_static Parquet::parquet_static OpenMP::OpenMP_CXX)
set_target_properties(ipopt_scan PROPERTIES 
                                    POSITION_INDEPENDENT_CODE ON
                                    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
