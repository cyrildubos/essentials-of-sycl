cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_COMPILER icpx)

project(essentials-oneapi LANGUAGES CXX)

find_package(IntelDPCPP REQUIRED)

add_executable(accessors_raw src/accessors_raw.cpp)
add_executable(accessors_war_waw src/accessors_war_waw.cpp)
add_executable(linear_buffers_graph src/linear_buffers_graph.cpp)
add_executable(linear_event_graph src/linear_event_graph.cpp)
add_executable(linear_in_order_graph src/linear_in_order_graph.cpp)
add_executable(task_scheduling src/task_scheduling.cpp)
add_executable(usm_explicit src/usm_explicit.cpp)
add_executable(usm_implicit src/usm_implicit.cpp)
add_executable(y_pattern_buffers_graph src/y_pattern_buffers_graph.cpp)
add_executable(y_pattern_event_graph src/y_pattern_event_graph.cpp)
add_executable(y_pattern_in_order_graph src/y_pattern_in_order_graph.cpp)