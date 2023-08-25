cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_COMPILER icpx)

project(essentials-oneapi LANGUAGES CXX)

find_package(IntelDPCPP REQUIRED)

add_executable(atomics_lab                              src/atomics_lab.cpp)
add_executable(local_memory_info                        src/local_memory_info.cpp)
add_executable(matrix_multiplication_16x16_local_memory src/matrix_multiplication_16x16_local_memory.cpp)
add_executable(matrix_multiplication_16x16              src/matrix_multiplication_16x16.cpp)
add_executable(reduction_atomics_buffer                 src/reduction_atomics_buffer.cpp)
add_executable(reduction_atomics_usm                    src/reduction_atomics_usm.cpp)