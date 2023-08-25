cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_COMPILER icpx)

project(essentials-oneapi LANGUAGES CXX)

find_package(IntelDPCPP REQUIRED)

add_executable(usm_data_1   src/usm_data_1.cpp)
add_executable(usm_data_2   src/usm_data_2.cpp)
add_executable(usm_explicit src/usm_explicit.cpp)
add_executable(usm_lab      src/usm_lab.cpp)
add_executable(usm          src/usm.cpp)
