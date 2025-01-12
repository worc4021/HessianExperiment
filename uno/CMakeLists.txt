add_executable(uno_scan src/main_uno.cpp)
target_include_directories(uno_scan PRIVATE ${GLOBAL_INCLUDES} include)
target_link_libraries(uno_scan PRIVATE autodiff::autodiff uno::uno Eigen3::Eigen Arrow::arrow_static Parquet::parquet_static OpenMP::OpenMP_CXX)
set_target_properties(uno_scan PROPERTIES 
                                POSITION_INDEPENDENT_CODE ON
                                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
                                CXX_STANDARD 20)