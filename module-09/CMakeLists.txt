cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_COMPILER icpx)

project(essentials-oneapi LANGUAGES CXX)

find_package(IntelDPCPP REQUIRED)

add_executable(accessors_sample         src/accessors_sample.cpp)
add_executable(buffer_creation          src/buffer_creation.cpp)
add_executable(buffer_set_final_data    src/buffer_set_final_data.cpp)
add_executable(buffer_set_write_back    src/buffer_set_write_back.cpp)
add_executable(host_accessor_init       src/host_accessor_init.cpp)
add_executable(host_accessor_sample     src/host_accessor_sample.cpp)
add_executable(lab_buffers              src/lab_buffers.cpp)
add_executable(sub_buffers              src/sub_buffers.cpp)

# add_executable(buffer_host_ptr src/buffer_host_ptr.cpp)
# add_executable(buffer_creation_uncommon src/buffer_creation_uncommon.cpp)