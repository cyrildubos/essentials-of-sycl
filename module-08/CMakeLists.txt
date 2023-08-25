cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_COMPILER icpx)

project(essentials-oneapi LANGUAGES CXX)

find_package(IntelDPCPP REQUIRED)

add_executable(multiple_reductions_buffers src/multiple_reductions_buffers.cpp)
add_executable(reduction_custom_operator   src/reduction_custom_operator.cpp)
add_executable(reduction_lab               src/reduction_lab.cpp)
add_executable(sum_reduction_buffers       src/sum_reduction_buffers.cpp)
add_executable(sum_reduction_usm           src/sum_reduction_usm.cpp)
add_executable(sum_single_task             src/sum_single_task.cpp)
add_executable(sum_workgroup_reduce        src/sum_workgroup_reduce.cpp)