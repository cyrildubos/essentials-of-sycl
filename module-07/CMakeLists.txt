cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_COMPILER icpx)

project(essentials-oneapi LANGUAGES CXX)

find_package(IntelDPCPP REQUIRED)

add_executable(dpl_buffer        src/dpl_buffer.cpp)
add_executable(dpl_simple        src/dpl_simple.cpp)
add_executable(dpl_sortdouble    src/dpl_sortdouble.cpp)
add_executable(dpl_usm_allocator src/dpl_usm_allocator.cpp)
add_executable(dpl_usm_pointer   src/dpl_usm_pointer.cpp)

# Parallel API