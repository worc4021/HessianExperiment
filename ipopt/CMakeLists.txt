
find_package(PkgConfig REQUIRED)
pkg_check_modules(ipopt REQUIRED IMPORTED_TARGET ipopt)

set(COINIPOPTLIBS)
if (WIN32)
    set(CMAKE_CXX_FLAGS "/Wall /EHsc")
    list(APPEND COINIPOPTLIBS ${ipopt_LIBRARY_DIRS}/ipopt.lib ${ipopt_LIBRARY_DIRS}/coinhsl.lib ${ipopt_LIBRARY_DIRS}/coinmumps.lib ${ipopt_LIBRARY_DIRS}/libcoinmetis.lib MKL::MKL)
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -fpie")
    list(APPEND COINIPOPTLIBS PkgConfig::ipopt)
endif()

add_executable(ipopt_scan src/main.cpp src/nlp.cpp)

target_include_directories(ipopt_scan PRIVATE 
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${ipopt_INCLUDE_DIRS}
                            ${GLOBAL_INCLUDES})

target_link_libraries(ipopt_scan PRIVATE ${COINIPOPTLIBS} autodiff::autodiff Arrow::arrow_static Parquet::parquet_static OpenMP::OpenMP_CXX)
set_target_properties(ipopt_scan PROPERTIES 
                                    POSITION_INDEPENDENT_CODE ON
                                    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
                                    CXX_STANDARD 20)
