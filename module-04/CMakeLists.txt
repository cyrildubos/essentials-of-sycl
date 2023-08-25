cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_COMPILER icpx)

project(essentials-oneapi LANGUAGES CXX)

find_package(IntelDPCPP REQUIRED)

add_executable(sub_group_broadcast src/sub_group_broadcast.cpp)
add_executable(sub_group_info      src/sub_group_info.cpp)
add_executable(sub_group_lab       src/sub_group_lab.cpp)
add_executable(sub_group_reduce    src/sub_group_reduce.cpp)
add_executable(sub_group_shuffle   src/sub_group_shuffle.cpp)
add_executable(sub_group_votes     src/sub_group_votes.cpp)

# add_executable(sub_group_reqd_size src/sub_group_reqd_size.cpp)
