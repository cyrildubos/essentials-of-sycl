cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_COMPILER icpx)

project(essentials-oneapi LANGUAGES CXX)

find_package(IntelDPCPP REQUIRED)

add_executable(buffer_destruction     src/buffer_destruction.cpp)
add_executable(buffer_sample          src/buffer_sample.cpp)
add_executable(gpu_sample             src/gpu_sample.cpp)
add_executable(host_accessor_sample   src/host_accessor_sample.cpp)
add_executable(vector_addition        src/vector_addition.cpp)

# add_executable(custom_device_sample src/custom_device_sample.cpp)
# add_executable(complex_multiplication src/complex_multiplication.cpp)